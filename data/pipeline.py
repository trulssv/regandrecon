"""
The data pipeline (see data/README.md): builds the simulated dataset from the quality assessment of data/triage.py.
1. Patient-grouped train-val-test split (make_splits).
2. Preprocessing of each study to attenuation values in mm^-1 (data.transforms.DataTransform).
3. Forward projection of each time bin with its own RayTransform geometry, giving clean (noise-free) line integrals.
   Noise is added on the fly by the quality loader (data.loaders.get_quality_loader with data.transforms.PoissonNoise).

Run: data/scripts/simulate.sh [--qualities high medium] [--geometry parallel3d|conebeam] [--overwrite]
"""
import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from data.config import DEFAULT_GEOMETRY, PREPROCESS_CFG_PATH, QUALITY_ASSESSMENT_PATH, RAY_TRAFO_CFG_PATHS, SIMULATED_ROOT
from data.loaders import load_study
from data.transforms import DataTransform
from data.utils import load_json, save_json_atomic
from operators.tomo.ray_trafo import RayTransform
from tracking import config_hash, file_hash, pin_config, start_run

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def make_splits(
        quality_assessment: dict,
        ratios: dict[str, float],
        seed: int = 42,
        existing: dict | None = None,
        n_time_bins: int | None = None,
        ) -> dict:
    """
    Patient-grouped train-val-test split of the assessed studies: all studies of a patient end up in the same split, regardless of their quality,
    so that no anatomy is shared between the splits.

    Patients are visited in a seeded random order and greedily assigned to the split that is furthest below its target number of studies.
    Patient assignments of an existing split manifest are kept, so that assessing more studies never moves an already assigned patient (and thereby
    never leaks a test patient into training); only new patients are assigned.

    input:
        quality_assessment: The content of quality_assessment.json, see assess_data_quality.
        ratios: The target fraction of studies per split, e.g. {"train": 0.7, "val": 0.15, "test": 0.15}.
        seed: The seed for the patient order.
        existing: A previous output of make_splits whose patient assignments are kept.
        n_time_bins: If given, studies with a different number of time bins are left out.
    output:
        The split manifest {"seed", "ratios", "patients": {patient_id: split}, "studies": {study_id: {"split", "quality", "patient_id", "study_dir"}}}.
    """

    studies = {}
    for study_id, entry in quality_assessment.items():
        if n_time_bins is not None and entry["n_time_bins"] != n_time_bins:
            logger.warning("Leaving out %s, expected %d time bins but got %d.", study_id, n_time_bins, entry["n_time_bins"])
            continue
        studies[study_id] = {
            "quality": entry["quality"],
            "patient_id": entry["scan_info"]["patient_id"],
            "study_dir": entry["study_dir"],
        }

    studies_per_patient: dict[str, int] = {}
    for study in studies.values():
        studies_per_patient[study["patient_id"]] = studies_per_patient.get(study["patient_id"], 0) + 1

    patients: dict[str, str] = dict(existing["patients"]) if existing is not None else {}
    counts = {split: 0 for split in SPLITS}
    for patient_id, split in patients.items():
        counts[split] += studies_per_patient.get(patient_id, 0)

    # Sort before shuffling so the order only depends on the seed and the set of new patients.
    new_patients = sorted(p for p in studies_per_patient if p not in patients)
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(new_patients), generator=g).tolist():
        patient_id = new_patients[idx]
        total = sum(counts.values()) + studies_per_patient[patient_id]
        split = max(SPLITS, key=lambda s: ratios[s] * total - counts[s])  # Ties are broken in the order train, val, test.
        patients[patient_id] = split
        counts[split] += studies_per_patient[patient_id]

    for study in studies.values():
        study["split"] = patients[study["patient_id"]]

    return {"seed": seed, "ratios": ratios, "patients": patients, "studies": studies}


def study_extent(scan_info: dict, shape: tuple[int, ...]) -> tuple[float, float, float]:
    """Physical (D, H, W) extent in mm of a raw volume with the given shape. The extent is unchanged by resizing the volume."""
    d, h, w = shape
    pixel_spacing = scan_info["resampled_pixel_spacing"]  # (W spacing, H spacing), mm
    slice_thickness = scan_info["resampled_slice_thickness"]  # mm
    return (d * slice_thickness, h * pixel_spacing[1], w * pixel_spacing[0])


def simulate_study(study_dir: Path, transform: DataTransform, ray_cfg: dict, time_step: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Preprocesses and forward projects one raw study.

    output:
        volume: (T, D, H, W) attenuation values in mm^-1.
        sinogram: (T, views, cols, rows) clean line integrals, one geometry per time bin.
        meta: The scan info with the voxel spacing of the preprocessed grid (spacing * shape = extent), plus raw_scan_info, raw_shape,
              shape, extent, n_time_bins and time_steps.
    """
    series = load_study(study_dir)
    raw_shape = tuple(series.data[0].shape)

    # NOTE: All studies currently share the same physical extent, but the extent (and thereby the ray transform) is kept study specific.
    # TODO: Batching studies with a single ray transform (batch_size > 1) relies on equal extents; resample to a fixed extent if this changes.
    extent = study_extent(series.scan_info, raw_shape)

    # Preprocess: (T, D, H, W) in the normalized range -> resized (T, D, H, W) attenuation values in mm^-1

    x = transform(torch.stack(series.data)).cpu()
    n_time_bins = x.shape[0]
    shape = tuple(x.shape[1:])

    # Simulate: one geometry per time bin, each lasting time_step time units (no gating times are available).

    time_steps = [time_step] * n_time_bins
    ray_trafo = RayTransform(**ray_cfg, extent=extent, shape=shape)
    ray_trafo._init_ray_transform(shape=shape, extent=extent, time_steps=time_steps)

    with torch.no_grad():
        y = ray_trafo([xt.unsqueeze(0) for xt in x])  # list of T (1, views, cols, rows)
    y = torch.cat(y, dim=0)  # (T, views, cols, rows)

    d, h, w = shape
    meta = {
        **series.scan_info,
        "resampled_pixel_spacing": [extent[2] / w, extent[1] / h],  # (W spacing, H spacing), mm
        "resampled_slice_thickness": extent[0] / d,  # mm
        "raw_scan_info": series.scan_info,
        "raw_shape": raw_shape,
        "shape": shape,
        "extent": extent,
        "n_time_bins": n_time_bins,
        "time_steps": time_steps,
    }
    return x, y, meta


def data_pipeline(qualities: tuple[str, ...] = ("high",), geometry: str = DEFAULT_GEOMETRY, out_root: Path | None = None, overwrite: bool = False,
                  quality_assessment_path: Path = QUALITY_ASSESSMENT_PATH) -> None:
    """
    Builds the simulated dataset: splits the assessed studies, then preprocesses and simulates each study of the given qualities.

    Each study is saved to out_root/<quality>/<split>/<study_id>/ as volume.pt (T, D, H, W), sinogram.pt (T, views, cols, rows) and meta.json.
    Studies that are already done are skipped unless overwrite is set, so an interrupted run can simply be restarted.

    Bookkeeping in out_root (see tracking/run.py):
        config.json   the simulation config (preprocessing, ray transform, time step). An out_root only ever holds data of a single config.
        splits.json   the patient-grouped split. Patient assignments are kept when more studies are assessed.
        runs.jsonl    one record per run: config hash, git commit, package versions, args and results (which studies were simulated).
        logs/         the log output of each run.
    Each study's meta.json records the config hash and run id that produced it.

    input:
        qualities: The qualities from the quality assessment to simulate.
        geometry: The ray transform config to simulate with, one of RAY_TRAFO_CFG_PATHS (data/config.py).
        out_root: The output directory. Defaults to SIMULATED_ROOT / geometry.
        overwrite: Re-simulate studies that are already done.
        quality_assessment_path: The quality assessment written by data/triage.py.
    """

    if out_root is None:
        out_root = SIMULATED_ROOT / geometry

    preprocess_cfg = load_json(PREPROCESS_CFG_PATH)
    config = {
        "geometry": geometry,
        "preprocess": {k: v for k, v in preprocess_cfg.items() if k != "train_val_test_split"},
        "ray_trafo": load_json(RAY_TRAFO_CFG_PATHS[geometry]),
        "time_step": 1.0,
    }
    pin_config(out_root / "config.json", config)  # Refuse to mix data simulated with different configs in the same out_root.

    args = {"qualities": list(qualities), "out_root": out_root, "overwrite": overwrite, "quality_assessment_path": quality_assessment_path}
    with start_run("simulation", config, log_dir=out_root, args=args) as run:
        quality_assessment = load_json(quality_assessment_path)

        # Patient-grouped train-val-test split. The split ratios and seed are recorded in splits.json.

        splits_path = out_root / "splits.json"
        split_cfg = preprocess_cfg["train_val_test_split"]
        splits = make_splits(
            quality_assessment,
            ratios={split: split_cfg[split] for split in SPLITS},
            seed=split_cfg.get("random_seed", 42),
            existing=load_json(splits_path) if splits_path.exists() else None,
            n_time_bins=preprocess_cfg["time_bins"],
        )
        save_json_atomic(splits_path, splits)
        run.results.update({"quality_assessment_hash": file_hash(quality_assessment_path), "splits_hash": config_hash(splits), "simulated": [], "skipped": []})

        transform = DataTransform.from_config(preprocess_cfg)

        studies = {study_id: info for study_id, info in splits["studies"].items() if info["quality"] in qualities}
        logger.info("Simulating %d studies of qualities %s into %s", len(studies), list(qualities), out_root)

        for study_id, info in tqdm(studies.items(), desc="Simulating studies"):
            study_out = out_root / info["quality"] / info["split"] / study_id
            if (study_out / "done").exists() and not overwrite:
                run.results["skipped"].append(study_id)
                continue

            x, y, meta = simulate_study(Path(info["study_dir"]), transform, config["ray_trafo"], config["time_step"])
            meta.update({"study_id": study_id, "quality": info["quality"], "split": info["split"], "geometry": geometry,
                         "config_hash": run.config_hash, "run_id": run.run_id})

            study_out.mkdir(parents=True, exist_ok=True)
            (study_out / "done").unlink(missing_ok=True)
            torch.save(x, study_out / "volume.pt")
            torch.save(y, study_out / "sinogram.pt")
            save_json_atomic(study_out / "meta.json", meta)
            (study_out / "done").touch()

            run.results["simulated"].append(study_id)
            logger.info("Simulated %s (%s/%s): volume %s, sinogram %s", study_id, info["quality"], info["split"], tuple(x.shape), tuple(y.shape))

        logger.info("Simulated %d studies, skipped %d already done.", len(run.results["simulated"]), len(run.results["skipped"]))


def main():
    parser = argparse.ArgumentParser(description="Split, preprocess and simulate projection data for the assessed studies.")
    parser.add_argument("--qualities", nargs="+", default=["high", "medium"], choices=["high", "medium", "low"])
    parser.add_argument("--geometry", default=DEFAULT_GEOMETRY, choices=sorted(RAY_TRAFO_CFG_PATHS))
    parser.add_argument("--out-root", type=Path, default=None, help="Defaults to SIMULATED_ROOT/<geometry>.")
    parser.add_argument("--overwrite", action="store_true", help="Re-simulate studies that are already done.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data_pipeline(qualities=tuple(args.qualities), geometry=args.geometry, out_root=args.out_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
