"""
Histograms of the physical extent of all raw 4D CT studies (see data/extent.py for what is measured), against the detector of the
parallel3d ray transform config and the study_filter of configs/preprocess.json. Used to choose a detector geometry that matches the
object instead of leaving most of the scanner's field of view unused.

Reads the cached extents (data/study_extents.json, written by data/scripts/extents.sh), or measures them with --remeasure.
Saves the per-study measurements as a csv and the histograms as a png in PLOT_DIR, and prints a summary.

Run directly: python test/plot_volume_extent.py [--remeasure] [--workers 16]
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.config import DATA_ROOT, PREPROCESS_CFG_PATH, RAY_TRAFO_CFG_PATHS, STUDY_EXTENTS_PATH
from data.extent import compute_study_extents, filter_reason, load_study_extents
from data.utils import load_json, save_json_atomic

PLOT_DIR = Path("test/plots/volume_extent")

PREPROCESS_CFG = load_json(PREPROCESS_CFG_PATH)
STUDY_FILTER = PREPROCESS_CFG["study_filter"]
RAY_CFG = load_json(RAY_TRAFO_CFG_PATHS["parallel3d"])

# Colors of the reference palette (light mode).
GRID_COLOR, OBJECT_COLOR = "#2a78d6", "#eb6834"
TEXT_SECONDARY, GRID_LINE = "#52514e", "#e4e3df"


def _hist(ax, series: list[tuple[str, np.ndarray, str]], limits: list[tuple[str, float, str]], title: str) -> None:
    values = np.concatenate([v for _, v, _ in series] + [[x for _, x, _ in limits]])
    bins = np.linspace(0.0, values.max() * 1.05, 50)
    for label, v, color in series:
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.35, color=color, linewidth=0)
        ax.hist(v, bins=bins, histtype="step", color=color, linewidth=2, label=f"{label} (max {v.max():.0f} mm)")
    for label, x, linestyle in limits:
        ax.axvline(x, color=TEXT_SECONDARY, linestyle=linestyle, linewidth=1.5, label=f"{label} ({x:.0f} mm)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("mm", color=TEXT_SECONDARY)
    ax.set_ylabel("studies", color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID_LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def plot(rows: list[dict], path: Path) -> None:
    col = lambda key: np.array([r[key] for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    _hist(axes[0], [("grid diagonal", col("grid_diameter"), GRID_COLOR), ("object 2·r_max", col("object_diameter"), OBJECT_COLOR)],
          [("detector", RAY_CFG["DetectorColExtent"], "--"), ("filter", STUDY_FILTER["max_fov_diameter_mm"], ":")],
          "Detector columns: in-plane FOV diameter")
    _hist(axes[1], [("grid", col("grid_d"), GRID_COLOR), ("object", col("object_d"), OBJECT_COLOR)],
          [("detector", RAY_CFG["DetectorRowExtent"], "--"), ("filter", STUDY_FILTER["max_axial_extent_mm"], ":")],
          "Detector rows: axial extent (D)")
    _hist(axes[2], [("object H", col("object_h"), GRID_COLOR), ("object W", col("object_w"), OBJECT_COLOR)], [],
          "Object bounding box in-plane (H, W)")
    n_kept = sum(filter_reason(r, STUDY_FILTER) is None for r in rows)
    fig.suptitle(f"Physical extent of {len(rows)} studies, {n_kept} pass the study filter "
                 f"(object: HU > {STUDY_FILTER['object_hu']:.0f}, union over gated phases)", x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(rows: list[dict]) -> None:
    print(f"\n{'quantity':<18}{'min':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}   (mm)")
    for key in ("grid_d", "grid_h", "grid_w", "grid_diameter", "object_d", "object_h", "object_w", "object_diameter"):
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"{key:<18}" + "".join(f"{x:8.1f}" for x in (v.min(), *np.percentile(v, [50, 95, 99]), v.max())))
    print(f"\nparallel3d detector: {RAY_CFG['DetectorColExtent']} mm x {RAY_CFG['DetectorRowExtent']} mm "
          f"({RAY_CFG['nDetectorCols']} cols x {RAY_CFG['nDetectorRows']} rows)")
    print(f"Untruncated object needs at least: {max(r['object_diameter'] for r in rows):.1f} mm (cols) x "
          f"{max(r['object_d'] for r in rows):.1f} mm (rows)")

    kept = [r for r in rows if filter_reason(r, STUDY_FILTER) is None]
    print(f"\nStudy filter {STUDY_FILTER}: keeps {len(kept)} of {len(rows)} studies")
    if kept:
        print(f"Kept studies need at least: {max(r['object_diameter'] for r in kept):.1f} mm (cols) x {max(r['object_d'] for r in kept):.1f} mm (rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remeasure", action="store_true", help=f"Measure the extents again and update {STUDY_EXTENTS_PATH}.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.remeasure or not STUDY_EXTENTS_PATH.exists():
        save_json_atomic(STUDY_EXTENTS_PATH, compute_study_extents(args.data_root, args.workers, PREPROCESS_CFG))
    rows = [{k: v for k, v in r.items() if k != "shape"} | dict(zip(("shape_d", "shape_h", "shape_w"), r["shape"]))
            for r in load_study_extents(PREPROCESS_CFG).values()]

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLOT_DIR / "extent.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, PLOT_DIR / "extent_histogram.png")
    summarize(rows)
    print(f"\nSaved {PLOT_DIR / 'extent.csv'} and {PLOT_DIR / 'extent_histogram.png'}")


if __name__ == "__main__":
    main()
