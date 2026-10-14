"""
Timeline of the key steps of a training step: wall time and GPU memory at the start and end of each labeled region.

region(label) replaces torch.profiler.record_function in the model, the loss and the training loop. It always emits the
record_function label (so torch.profiler still sees it), and while a StepTimeline is recording it also synchronizes the device
and samples the time and allocated memory at entry and exit. When no timeline is recording it is a plain record_function, so the
synchronization cost is only paid on the recorded steps.

Usage:
    timeline = StepTimeline(device)
    with timeline.record():                     # one training step
        with region("step/forward"): ...
        with region("step/backward"): ...
    timeline.events, timeline.samples           # the regions and memory samples of the last recorded step
    timeline.history                            # per recorded step, the total time of each key step (see key_step_times)

NOTE: With gradient checkpointing, the forward pass of each checkpointed block runs again during backward, so its regions
(hlpd/iter<k>, or hlpd/iter<k>/t<t> for each time step of the recurrent model, and hlpd/dual_net, ...) are recorded a second time
with in_backward=True. The start of the first recompute of iteration k marks where its backward pass starts, which is what splits
the backward pass into per iteration phases.
NOTE: Memory is torch.cuda.memory_allocated, i.e. the tensors held by torch. ASTRA (the ray transforms) allocates its own device
memory outside of torch, which only shows up in device_used_gb (cudaMemGetInfo, which also counts the caching allocator's reserve).
"""
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.profiler import record_function

_ACTIVE: "StepTimeline | None" = None   # the timeline that is currently recording, if any

ITER_LABEL = re.compile(r"^hlpd/iter(\d+)$")
RECOMPUTE_LABEL = re.compile(r"^hlpd/iter(\d+)(?:/t\d+)?$")    # a checkpointed iteration, or a checkpointed time step of one


@dataclass
class MemorySample:
    time_s: float           # since the start of the step
    label: str              # region that was entered or exited
    edge: str               # "enter" or "exit"
    allocated_gb: float     # torch.cuda.memory_allocated at the sample
    peak_gb: float          # max allocated since the previous sample, i.e. the peak between two samples
    device_used_gb: float   # total - free device memory, includes the allocator cache, ASTRA and other processes


@dataclass
class RegionEvent:
    label: str
    parents: tuple[str, ...]  # labels of the enclosing regions, outermost first
    start_s: float
    end_s: float
    in_backward: bool         # recorded during loss.backward(), i.e. a recompute of a checkpointed block

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@contextmanager
def region(label: str) -> Iterator[None]:
    """A record_function region that is also sampled by the recording StepTimeline, if any."""
    with record_function(label):
        timeline = _ACTIVE
        if timeline is None:
            yield
            return
        timeline._enter(label)
        try:
            yield
        finally:
            timeline._exit(label)


class StepTimeline:
    """Records the labeled regions of one training step at a time, see the module docstring."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.samples: list[MemorySample] = []
        self.events: list[RegionEvent] = []
        self.history: list[dict[str, float]] = []   # key_step_times of each recorded step
        self.peaks: dict[str, float] = {}           # peak allocated and reserved memory over the last recorded step, as in StepStats
        self._stack: list[tuple[str, float]] = []
        self._t0 = 0.0

    @contextmanager
    def record(self) -> Iterator["StepTimeline"]:
        global _ACTIVE
        assert _ACTIVE is None, "Another StepTimeline is already recording."
        self.samples, self.events, self._stack = [], [], []
        self.peaks = {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()
        _ACTIVE = self
        try:
            yield self
        finally:
            _ACTIVE = None
        self.history.append(key_step_times(self.events))

    def _sample(self, label: str, edge: str) -> float:
        # NOTE: the backward pass runs on autograd's own thread while the main thread waits in backward(), so a single stack is enough.
        if self.cuda:
            torch.cuda.synchronize(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            peak = torch.cuda.max_memory_allocated(self.device) / 1e9
            self.peaks["peak_allocated_gb"] = max(self.peaks["peak_allocated_gb"], peak)
            self.peaks["peak_reserved_gb"] = max(self.peaks["peak_reserved_gb"], torch.cuda.max_memory_reserved(self.device) / 1e9)
            torch.cuda.reset_peak_memory_stats(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            device_used = (total - free) / 1e9
        else:
            allocated = peak = device_used = 0.0
        t = time.perf_counter() - self._t0
        self.samples.append(MemorySample(t, label, edge, allocated, max(peak, allocated), device_used))
        return t

    def _enter(self, label: str) -> None:
        self._stack.append((label, self._sample(label, "enter")))

    def _exit(self, label: str) -> None:
        end = self._sample(label, "exit")
        stack_label, start = self._stack.pop()
        assert stack_label == label, f"Regions exited out of order: expected {stack_label}, got {label}."
        parents = tuple(lbl for lbl, _ in self._stack)
        self.events.append(RegionEvent(label, parents, start, end, in_backward="step/backward" in parents))


def _recomputes(events: list[RegionEvent]) -> dict[int, list[RegionEvent]]:
    """The recomputed (checkpointed) regions during backward per iteration: the whole iteration, or each of its time steps."""
    out: dict[int, list[RegionEvent]] = defaultdict(list)
    for e in events:
        match = RECOMPUTE_LABEL.match(e.label)
        if e.in_backward and match:
            out[int(match.group(1))].append(e)
    return out


def phases(events: list[RegionEvent]) -> list[tuple[str, float, float]]:
    """Coarse phases (name, start_s, end_s) of a recorded step, in order: the forward pass of each unrolled iteration, the loss,
    the backward pass of the loss and of each iteration (split at the recompute of the checkpointed iterations) and the optimizer."""
    out: list[tuple[str, float, float]] = []
    for e in sorted(events, key=lambda e: e.start_s):
        match = ITER_LABEL.match(e.label)
        if match and not e.in_backward:
            out.append((f"fwd it{match.group(1)}", e.start_s, e.end_s))
        elif e.label == "step/loss":
            out.append(("loss", e.start_s, e.end_s))
        elif e.label == "step/optimizer":
            out.append(("optim", e.start_s, e.end_s))

    backward = next((e for e in events if e.label == "step/backward"), None)
    if backward is not None:
        # NOTE: the backward pass of an iteration starts with its first recompute, i.e. of the whole iteration, or of its last time step.
        recomputes = sorted((min(e.start_s for e in es), k) for k, es in _recomputes(events).items())
        starts = [backward.start_s] + [start for start, _ in recomputes]
        names = ["bwd loss"] + [f"bwd it{k}" for _, k in recomputes]
        if not recomputes:
            names = ["backward"]
        ends = starts[1:] + [backward.end_s]
        out += list(zip(names, starts, ends))
    return sorted(out, key=lambda p: p[1])


# Key steps of the timing plot, in execution order. Forward regions are summed over time bins and iterations.
FORWARD_KEYS = ("hlpd/ray_trafo", "hlpd/dual_net", "hlpd/ray_trafo_adjoint", "hlpd/primal_net", "hlpd/motion_net",
                "loss/deform", "loss/ray_trafo", "loss/helmholtz")


def key_step_times(events: list[RegionEvent]) -> dict[str, float]:
    """Total time in seconds of each key step of a recorded step. Forward regions are keyed by their label, the backward phases
    by their phase name ("bwd loss", "bwd it3", ...), with the recompute part of each backward iteration under "<phase> recompute"."""
    times: dict[str, float] = defaultdict(float)
    for e in events:
        if e.label in FORWARD_KEYS and not e.in_backward:
            times[e.label] += e.duration_s
    recomputes = _recomputes(events)
    for name, start, end in phases(events):
        if name.startswith("bwd") or name in ("backward", "optim"):
            times[name] += end - start
            match = re.match(r"^bwd it(\d+)$", name)
            if match:
                times[f"{name} recompute"] += sum(e.duration_s for e in recomputes[int(match.group(1))])
    return dict(times)
