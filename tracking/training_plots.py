"""
Plots that are rewritten during training, so that a run can be followed while it is running (VS Code's image preview reloads them):
    gpu_memory.png      GPU memory over one training step, with the forward/backward phases labeled on the x-axis (see tracking/timeline.py)
    step_times.png      mean time of the key steps of the forward and backward pass over the recorded steps
    loss.png            the losses and the image domain error over the training steps
    prediction.gif      ground truth, deformed reconstruction and velocity field (RGB) in the axial/coronal/sagittal mid planes over time

Every file is written to a temporary file first and then moved into place, so a viewer never reads a half written file.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from tracking.timeline import FORWARD_KEYS, MemorySample, RegionEvent, phases

FORWARD_COLOR = "#4C72B0"
BACKWARD_COLOR = "#DD8452"
OTHER_COLOR = "#8C8C8C"


def _save_atomic(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)


def plot_gpu_memory(samples: list[MemorySample], events: list[RegionEvent], path: Path, step: int, device_total_gb: float | None = None) -> None:
    """Allocated GPU memory over one training step. The x-axis is labeled with the phases of the step (forward pass of each unrolled
    iteration, loss, backward pass of the loss and of each iteration, optimizer), the time in seconds is on the top axis."""
    t = np.array([s.time_s for s in samples])
    allocated = np.array([s.allocated_gb for s in samples])
    peak = np.array([s.peak_gb for s in samples])
    device_used = np.array([s.device_used_gb for s in samples])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    step_phases = phases(events)
    for i, (name, start, end) in enumerate(step_phases):
        color = FORWARD_COLOR if name.startswith("fwd") else BACKWARD_COLOR if name.startswith(("bwd", "backward")) else OTHER_COLOR
        ax.axvspan(start, end, color=color, alpha=0.08 if i % 2 else 0.16, lw=0)
    # Consecutive short phases (e.g. the loss and its backward pass) share one tick, so that their labels do not overlap.
    ticks: list[tuple[list[str], float, float]] = []
    for name, start, end in step_phases:
        short = end - start < 0.04 * t[-1]
        if ticks and short and ticks[-1][2] - ticks[-1][1] < 0.04 * t[-1]:
            ticks[-1] = (ticks[-1][0] + [name], ticks[-1][1], end)
        else:
            ticks.append(([name], start, end))
    ax.set_xticks([(start + end) / 2 for _, start, end in ticks], [" + ".join(names) for names, _, _ in ticks], rotation=30, ha="right", fontsize=8)

    # The peak between two samples belongs to the interval ending at the later sample, hence where="pre".
    ax.fill_between(t, allocated, peak, step="pre", color=FORWARD_COLOR, alpha=0.25, lw=0, label="peak between samples")
    # Memory is only known at the samples, so it is drawn as a step function rather than interpolated between them.
    ax.step(t, allocated, where="post", color=FORWARD_COLOR, lw=1.2, label="allocated (torch)")
    ax.step(t, device_used, where="post", color="k", lw=0.8, ls="--", label="device used (incl. cache, ASTRA)")
    if device_total_gb is not None:
        ax.axhline(device_total_gb, color="r", lw=0.8, ls=":", label=f"device total {device_total_gb:.0f} GB")

    i_peak = int(np.argmax(peak))
    phase_at_peak = next((name for name, start, end in step_phases if start <= t[i_peak] <= end), "")
    ax.annotate(f"peak {peak[i_peak]:.1f} GB\n{phase_at_peak}: {samples[i_peak].label}", xy=(t[i_peak], peak[i_peak]),
                xytext=(10, -28), textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.6})

    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, None)
    ax.set_ylabel("GPU memory (GB)")
    ax.secondary_xaxis("top").set_xlabel("time in step (s)", fontsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"GPU memory during step {step}", fontsize=10)
    _save_atomic(fig, path)


def _key_order(history: list[dict[str, float]]) -> list[str]:
    keys = set().union(*history)
    backward = sorted((k for k in keys if k.startswith("bwd it") and not k.endswith("recompute")), key=lambda k: -int(k.split("it")[1]))
    order = [k for k in FORWARD_KEYS if k in keys] + [k for k in ("backward", "bwd loss") if k in keys] + backward + (["optim"] if "optim" in keys else [])
    return order


def plot_step_times(history: list[dict[str, float]], path: Path) -> None:
    """Mean time per training step of the key steps, in execution order, over the recorded steps (error bars: std). Forward regions
    are summed over time bins and iterations. For the backward pass of each iteration, the hatched part is the recompute of the
    checkpointed forward pass."""
    if not history:
        return
    keys = _key_order(history)
    values = np.array([[h.get(k, np.nan) for k in keys] for h in history])
    mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    recompute = np.array([np.nanmean([h.get(f"{k} recompute", np.nan) for h in history]) if k.startswith("bwd it") else 0.0 for k in keys])

    colors = [FORWARD_COLOR if k in FORWARD_KEYS else OTHER_COLOR if k == "optim" else BACKWARD_COLOR for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(keys)), 4.5))
    ax.bar(x, mean, yerr=std if len(history) > 1 else None, color=colors, capsize=3, error_kw={"lw": 0.8})
    ax.bar(x, np.nan_to_num(recompute), fill=False, hatch="///", edgecolor="white", lw=0)
    for xi, m in zip(x, mean):
        ax.text(xi, m, f"{m * 1e3:.0f} ms" if m < 1 else f"{m:.2f} s", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x, [k.removeprefix("hlpd/") for k in keys], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("time per step (s)")
    total = np.nansum(mean)
    ax.set_title(f"Mean time of key steps over {len(history)} recorded steps (sum {total:.2f} s)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (FORWARD_COLOR, BACKWARD_COLOR, OTHER_COLOR)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=BACKWARD_COLOR, hatch="///", edgecolor="white"))
    ax.legend(handles, ["forward", "backward", "optimizer", "of which recompute"], fontsize=8, frameon=False)
    _save_atomic(fig, path)


def plot_losses(history: list[dict[str, float]], path: Path, zero_prediction_mse: float | None = None, val_history: list[dict[str, float]] | None = None) -> None:
    """The losses (log scale) and the image domain error of the deformed sequence over the training steps. val_history, if given,
    holds the validation metrics (same keys, with the step they were computed at) and is drawn as markers on top."""
    steps = [h["step"] for h in history]
    fig, (ax_loss, ax_mse) = plt.subplots(1, 2, figsize=(12, 4))
    for key, color in (("total_loss", "k"), ("data_loss", FORWARD_COLOR), ("regularization_loss", BACKWARD_COLOR)):
        ax_loss.plot(steps, [h[key] for h in history], color=color, lw=1, label=key + (" (unweighted)" if key == "regularization_loss" else ""))
        if val_history:
            ax_loss.plot([h["step"] for h in val_history], [h[key] for h in val_history], color=color, ls="none", marker="o", ms=4, mfc="white")
    if val_history:
        ax_loss.plot([], [], color="k", ls="none", marker="o", ms=4, mfc="white", label="validation")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("step")
    ax_loss.set_title("Loss", fontsize=10)
    ax_loss.legend(fontsize=8, frameon=False)

    ax_mse.plot(steps, [h["image_mse"] for h in history], color=FORWARD_COLOR, lw=1, label="image mse of deformed sequence")
    if val_history:
        ax_mse.plot([h["step"] for h in val_history], [h["image_mse"] for h in val_history], color=FORWARD_COLOR, ls="none", marker="o", ms=4, mfc="white", label="validation")
    if zero_prediction_mse is not None:
        ax_mse.axhline(zero_prediction_mse, color=OTHER_COLOR, ls="--", lw=0.8, label="zero prediction")
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("step")
    ax_mse.set_title(f"Image domain error (step {steps[-1]}: {history[-1]['image_mse']:.3g})", fontsize=10)
    ax_mse.legend(fontsize=8, frameon=False)
    for ax in (ax_loss, ax_mse):
        ax.grid(alpha=0.3, which="both", lw=0.5)
    _save_atomic(fig, path)


# Mid plane of a (D, H, W) volume: the fixed axis, and the (row, col) axes with their physical extent index into (D, H, W).
PLANES = {"axial": (0, (1, 2)), "coronal": (1, (0, 2)), "sagittal": (2, (0, 1))}


def _mid_slice(vol: np.ndarray, plane: str) -> np.ndarray:
    axis, _ = PLANES[plane]
    return np.take(vol, vol.shape[axis] // 2, axis=axis)


def plot_prediction_gif(
    x_pred: list[torch.Tensor],
    x_true: list[torch.Tensor],
    v: list[torch.Tensor],
    extent: tuple[float, float, float],
    path: Path,
    step: int,
    fps: int = 3,
) -> None:
    """GIF over the time bins of the ground truth, the deformed reconstruction and the velocity field, in the axial, coronal and
    sagittal mid planes (first batch element). x_pred and x_true are T tensors (B, D, H, W) and v is T tensors (B, D, H, W, 3) with
    grid_sample's (x, y, z) = (W, H, D) channel order, see FlowDeformationOperator. The velocity is shown as RGB = (x, y, z), mapped
    symmetrically around gray (zero velocity) with a common scale for all time bins."""
    planes = list(PLANES)
    true = [xt[0].detach().float().cpu().numpy() for xt in x_true]
    pred = [xp[0].detach().float().cpu().numpy().reshape(true[0].shape) for xp in x_pred]
    vel = [vt[0].detach().float().cpu().numpy() for vt in v]

    vmin, vmax = min(t.min() for t in true), max(t.max() for t in true)
    v_scale = float(np.percentile(np.abs(np.stack(vel)), 99)) or 1.0

    def rgb(vt: np.ndarray, plane: str) -> np.ndarray:
        return np.clip(0.5 + 0.5 * _mid_slice(vt, plane) / v_scale, 0.0, 1.0)

    rows = [("ground truth", lambda t, p: _mid_slice(true[t], p)), ("prediction", lambda t, p: _mid_slice(pred[t], p)), ("velocity", lambda t, p: rgb(vel[t], p))]
    fig, axes = plt.subplots(len(rows), len(planes), figsize=(3.5 * len(planes), 3.0 * len(rows)), dpi=100, squeeze=False)
    # NOTE: the figure size must be a whole number of pixels (3.4 inch * 3 * 100 dpi rounds down to 1019 pixels), and the layout fixed,
    # otherwise PillowWriter reads the frames with the wrong row stride and renders them sheared.
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.92, wspace=0.05, hspace=0.08)
    ims = []
    for r, (row_name, frame) in enumerate(rows):
        for c, plane in enumerate(planes):
            _, (row_axis, col_axis) = PLANES[plane]
            ax = axes[r, c]
            im = ax.imshow(frame(0, plane), cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=(0, extent[col_axis], 0, extent[row_axis]))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(plane, fontsize=10)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=10)
            ims.append((im, frame, plane))
    fig.text(0.5, 0.015, f"velocity: R, G, B = x (W), y (H), z (D) component, gray = 0, saturated at ±{v_scale:.3g}", ha="center", fontsize=8)
    title = fig.suptitle("", fontsize=11)

    def update(t: int):
        for im, frame, plane in ims:
            im.set_data(frame(t, plane))
        title.set_text(f"Step {step}, time bin {t + 1}/{len(true)}")
        return [im for im, _, _ in ims] + [title]

    update(0)
    anim = FuncAnimation(fig, update, frames=len(true), interval=1000 / fps, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    anim.save(tmp, writer=PillowWriter(fps=fps))
    plt.close(fig)
    os.replace(tmp, path)
