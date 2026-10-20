"""
Profiling helpers for training runs.

StepStats measures each training step: wall time, peak GPU memory, and GPU utilization (when pynvml is installed). It is cheap
enough to run on every step and its output can be passed straight to RunRecord.log_metrics.

make_profiler wraps torch.profiler for a few steps, and module_time_table summarizes the time spent in the record_function
regions of the model and loss (labels starting with "hlpd/" and "loss/", see model/hlpd.py and model/hlpd_loss.py).
The labels are no-ops when no profiler is running, so the model itself needs no profiling specific code path.

NOTE: With gradient checkpointing, the forward pass of each checkpointed block runs again during backward, so the hlpd/ regions
are counted twice per step.
NOTE: The ray transforms run through odl/astra, which synchronize with the host. Their CPU (wall) time is therefore the relevant
number, while the device time only covers the kernels launched by torch.
"""
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


class StepStats:
    """Per-step wall time, peak GPU memory and GPU utilization. Call start() before and stop() after each step."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self._t0 = 0.0

    def start(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()

    def stop(self) -> dict[str, float]:
        if self.cuda:
            torch.cuda.synchronize(self.device)
        stats = {"step_time_s": time.perf_counter() - self._t0}
        if self.cuda:
            stats["peak_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            stats["peak_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / 1e9
            try:
                # NOTE: a sample of the utilization at the end of the step, not an average over the step. Needs pynvml.
                stats["gpu_utilization_pct"] = float(torch.cuda.utilization(self.device))
            except Exception:
                pass
        return stats


def make_profiler(out_dir: Path, wait: int = 1, warmup: int = 1, active: int = 3) -> profile:
    """torch.profiler that skips `wait` steps, warms up for `warmup` steps and records `active` steps. Call prof.step() after each
    training step. The trace is written to out_dir and can be opened with TensorBoard or https://ui.perfetto.dev."""
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    return profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(out_dir)),
        profile_memory=True,
    )


def module_time_table(prof: profile, prefixes: tuple[str, ...] = ("hlpd/", "loss/", "step/")) -> str:
    """Table of the total and per-call time of the record_function regions whose label starts with one of the prefixes."""
    rows = []
    for event in prof.key_averages():
        if not event.key.startswith(prefixes):
            continue
        device_time = getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))
        rows.append((event.key, event.count, event.cpu_time_total / 1e3, device_time / 1e3))

    rows.sort(key=lambda row: row[2], reverse=True)
    lines = [f"{'region':<26} {'calls':>6} {'cpu total ms':>13} {'cpu ms/call':>12} {'device total ms':>16}"]
    for key, count, cpu_ms, device_ms in rows:
        lines.append(f"{key:<26} {count:>6} {cpu_ms:>13.1f} {cpu_ms / max(count, 1):>12.2f} {device_ms:>16.1f}")
    return "\n".join(lines)
