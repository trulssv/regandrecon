"""
Graphical inspection of simulated studies (the output of data/pipeline.py under SIMULATED_ROOT / <geometry>). For each selected study it
saves to PLOT_DIR / <study_id>/:

  - slices.png:    one time bin. Rows: the volume, its FBP and FBP - volume (axial, coronal and sagittal mid-slices, in mm), and the
                   noisy projections (mid detector row, mid view and mid detector col, views labelled by their gantry angle).
  - time_bins.png: all time bins. Columns: the coronal mid-slice of the volume and the FBP, and the mid detector row of the projections.
  - study.gif:     the volume, FBP and projection rows of slices.png, animated over the time bins.

The volume and the FBP share the intensity window of the volume, and every panel keeps its range across the time bins, so motion is not masked by
per-frame rescaling. Studies without an FBP (simulated before the pipeline stored them) are shown without the FBP rows.

This is a visualization/smoke-test script, not a pass/fail correctness suite (see test_data_pipeline.py for that). The test is skipped if no
simulated data is available.

Run: python test/visualize_simulated_studies.py [--n 3] [--study-ids ID ...] [--qualities high medium] [--modes train val test] [--time-bin 0]
 or: pytest test/visualize_simulated_studies.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from data.config import DEFAULT_SIMULATED_ROOT

PLOT_DIR = Path("test/plots/simulated_studies")
QUALITIES, MODES = ["high", "medium", "low"], ["train", "val", "test"]
FPS = 2

# (name, axis of the (D, H, W) volume fixed at its middle index, (x, y) physical extent axes of the slice).
VOLUME_SLICES = [("axial", 0, (2, 1)), ("coronal", 1, (2, 0)), ("sagittal", 2, (1, 0))]
# (name, axis of the (views, cols, rows) projections fixed at its middle index, whether to transpose the slice for display, y label, x label).
# The mid view is transposed so detector rows (aligned with the rotation axis, i.e. the volume's D axis) run vertically, like a radiograph.
PROJECTION_SLICES = [("mid row", 2, False, "angle [deg]", "col [mm]"), ("mid view", 0, True, "row [mm]", "col [mm]"), ("mid col", 1, False, "angle [deg]", "row [mm]")]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_studies(data_root: Path, qualities: list[str], modes: list[str], study_ids: list[str] | None = None) -> list[Path]:
    """The finished study directories under data_root/<quality>/<mode>/, optionally only those of study_ids."""
    study_dirs = sorted(d for q in qualities for m in modes for d in (data_root / q / m).glob("*/") if (d / "done").exists())
    if study_ids is not None:
        missing = set(study_ids) - {d.name for d in study_dirs}
        assert not missing, f"Studies {sorted(missing)} are not finished studies of qualities {qualities} and modes {modes} under {data_root}."
        study_dirs = [d for d in study_dirs if d.name in study_ids]
    return study_dirs


def load_study(study_dir: Path, ray_cfg: dict) -> dict:
    """The volume, FBP (None if not stored) and sinogram of a study as numpy arrays, its meta data, and the angular range of each time bin."""
    with open(study_dir / "meta.json") as f:
        meta = json.load(f)
    fbp_path = study_dir / "fbp.pt"
    study = {
        "meta": meta,
        "volume": torch.load(study_dir / "volume.pt", map_location="cpu").numpy(),      # (T, D, H, W)
        "fbp": torch.load(fbp_path, map_location="cpu").numpy() if fbp_path.exists() else None,
        "sinogram": torch.load(study_dir / "sinogram.pt", map_location="cpu").numpy(),  # (T, views, cols, rows)
    }

    # The arc of each time bin, as in RayTransform._init_geometry.
    time_steps = np.asarray(meta["time_steps"], dtype=float)
    overlap = ray_cfg.get("AngularOverlap", 0.0)
    starts = np.concatenate([[0.0], np.cumsum(time_steps)[:-1]]) * ray_cfg["GantrySpeed"] - overlap
    stops = np.cumsum(time_steps) * ray_cfg["GantrySpeed"] + overlap
    study["angles"] = np.degrees(np.stack([starts, stops], axis=1))                     # (T, 2)
    return study


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def volume_panels(volume: np.ndarray, extent: list[float]) -> list[tuple[str, np.ndarray, tuple]]:
    """The (name, slice, imshow extent) of the mid-slices of a (D, H, W) volume, in physical coordinates (mm)."""
    panels = []
    for name, axis, (ex, ey) in VOLUME_SLICES:
        s = np.take(volume, volume.shape[axis] // 2, axis=axis)
        panels.append((name, s, (-extent[ex] / 2, extent[ex] / 2, -extent[ey] / 2, extent[ey] / 2)))
    return panels


def projection_panels(y: np.ndarray, angles: tuple[float, float], ray_cfg: dict) -> list[tuple[str, np.ndarray, tuple]]:
    """The (name, slice, imshow extent) of the mid-slices of (views, cols, rows) projections, with views in degrees and the detector in mm."""
    c, r = ray_cfg["DetectorColExtent"] / 2, ray_cfg["DetectorRowExtent"] / 2
    a0, a1 = angles
    extents = {"mid row": (-c, c, a0, a1), "mid view": (-c, c, -r, r), "mid col": (-r, r, a0, a1)}
    panels = []
    for name, axis, transpose, _, _ in PROJECTION_SLICES:
        s = np.take(y, y.shape[axis] // 2, axis=axis)
        if name == "mid view":
            name = f"view at {a0 + (y.shape[0] // 2 + 0.5) * (a1 - a0) / y.shape[0]:.0f} deg"
        panels.append((name, s.T if transpose else s, extents[PROJECTION_SLICES[len(panels)][0]]))
    return panels


def _window(volume: np.ndarray) -> tuple[float, float]:
    """A robust intensity window of a volume. The FBP is shown in the window of its volume, so that its under- and overshoots are clipped
    instead of washing out the contrast of both."""
    return tuple(np.percentile(volume.ravel()[:: max(volume.size // 2_000_000, 1)], [0.5, 99.5]).tolist())


def _rows(study: dict, t: int, ray_cfg: dict) -> list[tuple[str, list[tuple[str, np.ndarray, tuple]], dict]]:
    """The rows of slices.png (and study.gif) at time bin t: (row label, panels, imshow kwargs)."""
    extent = study["meta"]["extent"]
    vol_window = _window(study["volume"])
    rows = [("volume", volume_panels(study["volume"][t], extent), {"cmap": "gray", "vmin": vol_window[0], "vmax": vol_window[1]})]
    if study["fbp"] is not None:
        rows.append(("FBP", volume_panels(study["fbp"][t], extent), {"cmap": "gray", "vmin": vol_window[0], "vmax": vol_window[1]}))
    rows.append(("projections", projection_panels(study["sinogram"][t], tuple(study["angles"][t]), ray_cfg), {"cmap": "gray"}))
    return rows


def _projection_windows(study: dict, ray_cfg: dict) -> list[tuple[float, float]]:
    """One window per projection panel, shared across the time bins."""
    panels = [projection_panels(study["sinogram"][t], tuple(study["angles"][t]), ray_cfg) for t in range(len(study["sinogram"]))]
    return [(min(p[i][1].min() for p in panels), max(p[i][1].max() for p in panels)) for i in range(len(PROJECTION_SLICES))]


def _label_axes(ax, row: str, col: int) -> None:
    if row == "projections":
        _, _, _, ylabel, xlabel = PROJECTION_SLICES[col]
    else:
        ylabel, xlabel = "mm", "mm"
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_slices(study: dict, t: int, ray_cfg: dict, savepath: Path) -> Path:
    """slices.png: the volume, FBP, FBP - volume and projection mid-slices at time bin t."""
    rows = _rows(study, t, ray_cfg)
    if study["fbp"] is not None:
        diff = study["fbp"][t] - study["volume"][t]
        lim = float(np.percentile(np.abs(diff), 99.5))
        rows.insert(2, ("FBP - volume", volume_panels(diff, study["meta"]["extent"]), {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim}))
    proj_windows = _projection_windows(study, ray_cfg)

    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.6 * len(rows)), squeeze=False)
    for i, (row, panels, kwargs) in enumerate(rows):
        for j, (name, s, extent) in enumerate(panels):
            ax = axes[i, j]
            if row == "projections":
                kwargs = {**kwargs, "vmin": proj_windows[j][0], "vmax": proj_windows[j][1]}
            im = ax.imshow(s, origin="lower", extent=extent, aspect="equal" if row != "projections" or j == 1 else "auto", **kwargs)
            ax.set_title(f"{row}: {name}", fontsize=10)
            _label_axes(ax, row, j)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    fig.suptitle(f"{study['meta']['study_id']} ({study['meta']['quality']}/{study['meta']['split']}), time bin {t + 1}/{len(study['volume'])}, "
                 f"arc {study['angles'][t][0]:.0f}-{study['angles'][t][1]:.0f} deg. Volume and FBP in mm$^{{-1}}$, projections in line integrals.", fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=110)
    plt.close(fig)
    return savepath


def plot_time_bins(study: dict, ray_cfg: dict, savepath: Path) -> Path:
    """time_bins.png: per time bin (rows) the coronal mid-slice of the volume and FBP, and the mid detector row of the projections."""
    T, extent = len(study["volume"]), study["meta"]["extent"]
    vol_window = _window(study["volume"])
    proj_window = _projection_windows(study, ray_cfg)[0]
    columns = ["volume"] + (["FBP"] if study["fbp"] is not None else []) + ["projections"]

    fig, axes = plt.subplots(T, len(columns), figsize=(4 * len(columns), 1.9 * T), squeeze=False)
    for t in range(T):
        for j, col in enumerate(columns):
            ax = axes[t, j]
            if col == "projections":
                name, s, ext = projection_panels(study["sinogram"][t], tuple(study["angles"][t]), ray_cfg)[0]
                ax.imshow(s, origin="lower", extent=ext, aspect="auto", cmap="gray", vmin=proj_window[0], vmax=proj_window[1])
            else:
                name, s, ext = volume_panels((study["volume"] if col == "volume" else study["fbp"])[t], extent)[1]
                ax.imshow(s, origin="lower", extent=ext, aspect="equal", cmap="gray", vmin=vol_window[0], vmax=vol_window[1])
            if t == 0:
                ax.set_title(f"{col}: {name}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"t = {t + 1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{study['meta']['study_id']}: all time bins", fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=100)
    plt.close(fig)
    return savepath


def plot_gif(study: dict, ray_cfg: dict, savepath: Path, fps: int = FPS) -> Path:
    """study.gif: the volume, FBP and projection rows of slices.png, animated over the time bins."""
    T = len(study["volume"])
    frames = [_rows(study, t, ray_cfg) for t in range(T)]
    proj_windows = _projection_windows(study, ray_cfg)

    fig, axes = plt.subplots(len(frames[0]), 3, figsize=(13, 3.6 * len(frames[0])), squeeze=False)
    ims, titles = [], []
    for i, (row, panels, kwargs) in enumerate(frames[0]):
        for j, (name, s, extent) in enumerate(panels):
            ax = axes[i, j]
            if row == "projections":
                kwargs = {**kwargs, "vmin": proj_windows[j][0], "vmax": proj_windows[j][1]}
            ims.append(ax.imshow(s, origin="lower", extent=extent, aspect="equal" if row != "projections" or j == 1 else "auto", **kwargs))
            titles.append(ax.set_title(f"{row}: {name}", fontsize=10))
            _label_axes(ax, row, j)
            fig.colorbar(ims[-1], ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    suptitle = fig.suptitle("", fontsize=11)
    fig.tight_layout()

    def update(t):
        k = 0
        for row, panels, _ in frames[t]:
            for name, s, extent in panels:
                ims[k].set_data(s)
                ims[k].set_extent(extent)
                titles[k].set_text(f"{row}: {name}")
                k += 1
        suptitle.set_text(f"{study['meta']['study_id']}, time bin {t + 1}/{T}, arc {study['angles'][t][0]:.0f}-{study['angles'][t][1]:.0f} deg")
        return ims + titles + [suptitle]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    anim.save(savepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return savepath


def visualize_study(study_dir: Path, ray_cfg: dict, plot_dir: Path = PLOT_DIR, time_bin: int = 0, fps: int = FPS) -> list[Path]:
    """Saves slices.png, time_bins.png and study.gif of a study to plot_dir / <study_id>/ and returns their paths."""
    study = load_study(study_dir, ray_cfg)
    out = plot_dir / study_dir.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{study_dir.name}: volume {study['volume'].shape}, sinogram {study['sinogram'].shape} (T, views, cols, rows), "
          f"FBP {'stored' if study['fbp'] is not None else 'missing'} -> {out}")
    return [plot_slices(study, time_bin, ray_cfg, out / "slices.png"), plot_time_bins(study, ray_cfg, out / "time_bins.png"),
            plot_gif(study, ray_cfg, out / "study.gif", fps)]


def simulation_ray_cfg(data_root: Path) -> dict:
    """The ray transform config the data under data_root was simulated with (pinned by data/pipeline.py)."""
    with open(data_root / "config.json") as f:
        return json.load(f)["config"]["ray_trafo"]


@pytest.mark.skipif(not (DEFAULT_SIMULATED_ROOT / "config.json").exists(), reason=f"No simulated data in {DEFAULT_SIMULATED_ROOT}")
def test_visualize_simulated_study():
    study_dirs = find_studies(DEFAULT_SIMULATED_ROOT, QUALITIES, MODES)
    if not study_dirs:
        pytest.skip(f"No finished studies in {DEFAULT_SIMULATED_ROOT}.")
    for path in visualize_study(study_dirs[0], simulation_ray_cfg(DEFAULT_SIMULATED_ROOT)):
        assert path.exists() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_SIMULATED_ROOT, help="SIMULATED_ROOT / <geometry>.")
    parser.add_argument("--qualities", nargs="+", default=QUALITIES, choices=QUALITIES)
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--study-ids", nargs="+", default=None, help="Only these studies (overrides --n).")
    parser.add_argument("--n", type=int, default=3, help="The number of studies to visualize, spread evenly over the found studies.")
    parser.add_argument("--time-bin", type=int, default=0, help="The time bin of slices.png.")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    args = parser.parse_args()

    study_dirs = find_studies(args.data_root, args.qualities, args.modes, args.study_ids)
    if not study_dirs:
        print(f"No finished studies of qualities {args.qualities} and modes {args.modes} in {args.data_root}.")
        return
    if args.study_ids is None and args.n < len(study_dirs):
        study_dirs = [study_dirs[i] for i in np.linspace(0, len(study_dirs) - 1, args.n).round().astype(int)]

    ray_cfg = simulation_ray_cfg(args.data_root)
    for study_dir in study_dirs:
        visualize_study(study_dir, ray_cfg, args.plot_dir, args.time_bin, args.fps)


if __name__ == "__main__":
    main()
