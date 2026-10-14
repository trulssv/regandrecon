"""
Graphical inspection of the quality loader's spatiotemporal data: loads one simulated study through get_quality_loader and saves two gifs
that animate over the study's time bins, one frame per time bin:

  - <study_id>_volume.gif:   the axial, coronal and sagittal mid-slices of the (D, H, W) volume, side by side, in physical aspect ratio
                             (slicing and extent are delegated to visualization.static_visualization.StaticVisualization).
  - <study_id>_sinogram.gif: the three mid-slices of the noiseless (views, cols, rows) projection data, side by side:
                             the mid detector row (a classic views x cols sinogram), the mid view (a single cols x rows projection
                             image) and the mid detector col (views x rows).

Each gif uses one intensity range per panel across all time bins, so motion is not masked by per-frame rescaling.

This is a visualization/smoke-test script, not a pass/fail correctness suite (see test_data_pipeline.py for that). It is skipped if no
simulated data is available under DEFAULT_SIMULATED_ROOT.

Run: pytest test/test_quality_loader_visualization.py
 or: python test/test_quality_loader_visualization.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RegAndReconDataset, get_quality_loader

PLOT_DIR = Path("test/plots/quality_loader")
QUALITY, MODE = "high", "train"
FPS = 2

VOLUME_PLANES = ["axial", "coronal", "sagittal"]

# (title, axis of the (views, cols, rows) projection that is fixed at its middle index, whether to transpose the slice for display, y label, x label).
# The mid view is transposed so detector rows (aligned with the rotation axis, i.e. the volume's D axis) run vertically, like a radiograph.
PROJECTION_SLICES = [
    ("Mid row", 2, False, "views", "cols"),
    ("Mid view", 0, True, "rows", "cols"),
    ("Mid col", 1, False, "views", "rows"),
]


def load_quality_study(quality: str = QUALITY, mode: str = MODE, data_root: Path = DEFAULT_SIMULATED_ROOT) -> dict | None:
    """The first study of the quality loader in collate_time_list format (batch size 1, with volume, without noise), or None if there is none."""
    if len(RegAndReconDataset([quality], mode, data_root=data_root)) == 0:
        return None
    loader = get_quality_loader([quality], mode, batch_size=1, shuffle=False, load_volume=True, data_root=data_root)
    return next(iter(loader))


def _save_gif(fig, ims: list, frames: list[list[np.ndarray]], suptitle_prefix: str, savepath: Path, fps: int) -> Path:
    """Animate frames (a list over time of one 2D array per panel) on the given images and save the animation as a gif."""
    suptitle = fig.suptitle(f"{suptitle_prefix} (time bin 1/{len(frames)})", fontsize=11)
    fig.tight_layout()

    def update(t):
        for im, panel in zip(ims, frames[t]):
            im.set_data(panel)
        suptitle.set_text(f"{suptitle_prefix} (time bin {t + 1}/{len(frames)})")
        return ims + [suptitle]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    os.makedirs(savepath.parent, exist_ok=True)
    print("Saving gif to:", savepath)
    anim.save(savepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return savepath


def plot_volume_mid_slices_gif(volume: list[torch.Tensor], meta: dict, savepath: Path, fps: int = FPS) -> Path:
    """Animate the axial/coronal/sagittal mid-slices of a list over time of (B, D, H, W) volumes (first batch element) as a gif."""
    from visualization.static_visualization import StaticVisualization

    volume = [v[:1].float() for v in volume]
    viz = StaticVisualization(meta_data=meta, batch_idx=0)
    viz._init_shape(volume)

    frames = [[viz._extract_slice(volume, plane=plane, time_bin=t).numpy() for plane in VOLUME_PLANES] for t in range(len(volume))]
    vmins = [min(frame[i].min() for frame in frames) for i in range(len(VOLUME_PLANES))]
    vmaxs = [max(frame[i].max() for frame in frames) for i in range(len(VOLUME_PLANES))]

    fig, axes = plt.subplots(1, len(VOLUME_PLANES), figsize=(4 * len(VOLUME_PLANES), 4.5))
    ims = []
    for ax, plane, vmin, vmax in zip(axes, VOLUME_PLANES, vmins, vmaxs):
        viz.vmin, viz.vmax = vmin, vmax
        im, title = viz.visualize_plane(volume, ax, plane=plane, time_bin=0)
        title.set_text(plane.capitalize())
        title.set_fontsize(10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    return _save_gif(fig, ims, frames, f"{meta['study_id']} volume [mm$^{{-1}}$]", savepath, fps)


def plot_projection_mid_slices_gif(sinogram: list[torch.Tensor], study_id: str, savepath: Path, fps: int = FPS) -> Path:
    """Animate the three mid-slices (see PROJECTION_SLICES) of a list over time of (B, views, cols, rows) projections (first batch element) as a gif."""
    frames = []
    for y in sinogram:
        y = y[0].float()
        panels = []
        for _, axis, transpose, _, _ in PROJECTION_SLICES:
            s = y.select(axis, y.shape[axis] // 2)
            panels.append((s.T if transpose else s).numpy())
        frames.append(panels)
    vmins = [min(frame[i].min() for frame in frames) for i in range(len(PROJECTION_SLICES))]
    vmaxs = [max(frame[i].max() for frame in frames) for i in range(len(PROJECTION_SLICES))]

    fig, axes = plt.subplots(1, len(PROJECTION_SLICES), figsize=(4 * len(PROJECTION_SLICES), 4.5))
    ims = []
    for i, (ax, (name, _, _, ylabel, xlabel)) in enumerate(zip(axes, PROJECTION_SLICES)):
        im = ax.imshow(frames[0][i], cmap="gray", vmin=vmins[i], vmax=vmaxs[i], aspect="auto", origin="lower")
        ax.set_title(f"{name} ({ylabel} x {xlabel})", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    return _save_gif(fig, ims, frames, f"{study_id} projections", savepath, fps)


def visualize_quality_study(quality: str = QUALITY, mode: str = MODE, data_root: Path = DEFAULT_SIMULATED_ROOT, plot_dir: Path = PLOT_DIR) -> tuple[Path, Path] | None:
    """Load the first study of the quality loader and save its volume and projection gifs to plot_dir. Returns the gif paths, or None if there is no study."""
    batch = load_quality_study(quality, mode, data_root)
    if batch is None:
        return None

    meta, study_id = batch["meta"][0], batch["study_id"][0]
    print(f"Study {study_id}: {len(batch['volume'])} time bins, volume {tuple(batch['volume'][0].shape[1:])}, projections {tuple(batch['sinogram'][0].shape[1:])} (views, cols, rows)")

    volume_path = plot_volume_mid_slices_gif(batch["volume"], meta, plot_dir / f"{study_id}_volume.gif")
    sinogram_path = plot_projection_mid_slices_gif(batch["sinogram"], study_id, plot_dir / f"{study_id}_sinogram.gif")
    return volume_path, sinogram_path


@pytest.mark.skipif(not DEFAULT_SIMULATED_ROOT.exists(), reason=f"No simulated data in {DEFAULT_SIMULATED_ROOT}")
def test_quality_loader_visualization():
    paths = visualize_quality_study()
    if paths is None:
        pytest.skip(f"No simulated {QUALITY}/{MODE} studies.")
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


if __name__ == "__main__":
    if visualize_quality_study() is None:
        print(f"No simulated {QUALITY}/{MODE} studies in {DEFAULT_SIMULATED_ROOT}.")
