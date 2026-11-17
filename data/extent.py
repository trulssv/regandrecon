"""
Physical extent of the raw 4D CT studies, used to leave out studies that do not fit the scanner (see "study_filter" in configs/preprocess.json).

For every study two extents are measured:
- grid extent: the (D, H, W) extent in mm of the voxel grid (data.utils.study_extent), i.e. the reconstruction space.
- object extent: the support of everything above object_hu (body, couch, ...), taken over the union of all gated phases.
  Components smaller than min_component_voxels are dropped as noise. Rays that only pass through air add nothing to the
  sinogram, so the detector only has to cover the object, not the whole grid.

The reco space is centered at the rotation axis (D, see operators/tomo/ray_trafo.py). With the parallel3d geometry the detector thereby needs
- columns: an extent of object_diameter = 2 * r_max, where r_max is the largest distance of the object from the rotation axis in the (H, W) plane,
- rows: an extent of object_d, the object's extent along D.
For a cone beam geometry these are object-side extents and have to be scaled by the magnification (det_radius + source_radius) / source_radius.

The extents are cached in STUDY_EXTENTS_PATH, as they need every volume to be loaded. Run: data/scripts/extents.sh [--workers 16]
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from data.config import DATA_ROOT, PREPROCESS_CFG_PATH, STUDY_EXTENTS_PATH
from data.loaders import _gated_series_dirs, index_studies, study_id_from_dir
from data.utils import load_json, save_json_atomic, study_extent

logger = logging.getLogger(__name__)


def _normalized_threshold(hu: float, preprocess_cfg: dict) -> float:
    """HU value -> the normalized range of the raw volumes (INITIAL_RANGE maps linearly onto HU_RANGE)."""
    (x0, x1), (h0, h1) = preprocess_cfg["INITIAL_RANGE"], preprocess_cfg["HU_RANGE"]
    return x0 + (hu - h0) / (h1 - h0) * (x1 - x0)


def _drop_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    keep = np.bincount(labels.ravel()) >= min_voxels
    keep[0] = False
    return keep[labels]


def measure_study(study_dir: Path, object_hu: float, min_component_voxels: int, preprocess_cfg: dict) -> dict | None:
    """Grid and object extents (mm) of one raw study, or None if it holds no readable series."""
    threshold = _normalized_threshold(object_hu, preprocess_cfg)
    mask, scan_info = None, None
    for _, series_dir in _gated_series_dirs(study_dir):
        volume_path, scan_info_path = series_dir / "volume.pt", series_dir / "scan_info.json"
        if not volume_path.exists() or not scan_info_path.exists():
            continue
        phase_mask = (torch.load(volume_path, map_location="cpu") > threshold).numpy()
        mask = phase_mask if mask is None else mask | phase_mask
        if scan_info is None:
            scan_info = load_json(scan_info_path)
    if mask is None:
        return None

    mask = _drop_small_components(mask, min_component_voxels)
    shape = mask.shape
    grid = study_extent(scan_info, shape)
    spacing = [e / n for e, n in zip(grid, shape)]
    # Voxel centers in mm, relative to the grid center (= the rotation axis).
    centers = [(np.arange(n) + 0.5) * s - e / 2 for n, s, e in zip(shape, spacing, grid)]

    result = {
        "study_id": study_id_from_dir(study_dir),
        "shape": list(shape),
        "grid_d": grid[0], "grid_h": grid[1], "grid_w": grid[2],
        "grid_diameter": float(np.hypot(grid[1], grid[2])),  # In-plane FOV diameter needed to cover the whole grid.
    }
    if not mask.any():
        return result | {"object_d": 0.0, "object_h": 0.0, "object_w": 0.0, "object_diameter": 0.0}

    # Bounding box extent along each axis, from the outer voxel edges.
    for axis, name in enumerate("dhw"):
        occupied = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        result[f"object_{name}"] = float(centers[axis][occupied[-1]] - centers[axis][occupied[0]] + spacing[axis])

    # Largest distance of the object from the rotation axis, to the outer corner of the voxel.
    yy, xx = np.nonzero(mask.any(axis=0))
    r = np.hypot(np.abs(centers[1][yy]) + spacing[1] / 2, np.abs(centers[2][xx]) + spacing[2] / 2)
    result["object_diameter"] = float(2 * r.max())
    return result


def compute_study_extents(data_root: Path = DATA_ROOT, workers: int = 16, preprocess_cfg: dict | None = None) -> dict:
    """Measures all studies under data_root with the object definition of preprocess_cfg["study_filter"].
    Returns {"object_hu", "min_component_voxels", "studies": {study_id: extents}}."""
    preprocess_cfg = load_json(PREPROCESS_CFG_PATH) if preprocess_cfg is None else preprocess_cfg
    object_hu, min_voxels = preprocess_cfg["study_filter"]["object_hu"], preprocess_cfg["study_filter"]["min_component_voxels"]
    study_dirs = index_studies(data_root)
    logger.info("Measuring %d studies under %s with %d workers.", len(study_dirs), data_root, workers)
    measure = partial(measure_study, object_hu=object_hu, min_component_voxels=min_voxels, preprocess_cfg=preprocess_cfg)
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(measure, study_dirs, chunksize=4) if r is not None]
    return {"object_hu": object_hu, "min_component_voxels": min_voxels, "studies": {r["study_id"]: r for r in rows}}


def load_study_extents(preprocess_cfg: dict, path: Path = STUDY_EXTENTS_PATH) -> dict:
    """The cached study extents. Raises if they are missing or were measured with another object definition than preprocess_cfg["study_filter"]."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Measure the study extents first with data/scripts/extents.sh.")
    extents = load_json(path)
    study_filter = preprocess_cfg["study_filter"]
    if (extents["object_hu"], extents["min_component_voxels"]) != (study_filter["object_hu"], study_filter["min_component_voxels"]):
        raise ValueError(f"{path} was measured with object_hu={extents['object_hu']}, min_component_voxels={extents['min_component_voxels']}, "
                         f"but the study_filter in {PREPROCESS_CFG_PATH} differs. Remeasure with data/scripts/extents.sh.")
    return extents["studies"]


def filter_reason(extent: dict, study_filter: dict) -> str | None:
    """Why a study with the given extents is left out by study_filter (configs/preprocess.json), or None if it is kept."""
    if extent["object_d"] > study_filter["max_axial_extent_mm"]:
        return f"axial extent {extent['object_d']:.1f} mm > {study_filter['max_axial_extent_mm']} mm"
    if extent["object_diameter"] > study_filter["max_fov_diameter_mm"]:
        return f"FOV diameter {extent['object_diameter']:.1f} mm > {study_filter['max_fov_diameter_mm']} mm"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the physical extent of all raw studies and cache it in STUDY_EXTENTS_PATH.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=STUDY_EXTENTS_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    extents = compute_study_extents(args.data_root, args.workers)
    save_json_atomic(args.out, extents)
    logger.info("Saved the extents of %d studies to %s", len(extents["studies"]), args.out)


if __name__ == "__main__":
    main()
