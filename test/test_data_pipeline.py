"""
Checks for the data pipeline: patient-grouped splitting, preprocessing, noise model, collation, an end-to-end pipeline run on tiny
synthetic data, and (if the simulated data is available) the quality loader on real data.

Run: pytest test/test_data_pipeline.py
"""
import json
from pathlib import Path

import pytest
import torch

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RawDataSet, RegAndReconDataset, collate_time_list, get_quality_loader, study_id_from_dir
import data.pipeline as pipeline
from data.pipeline import data_pipeline, make_splits, study_extent
from data.transforms import DataTransform, HUTransform, InverseHUTransform, PoissonNoise, ResizeTransform
from tracking import ConfigMismatchError
from typing import Sized
RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def fake_quality_assessment(n_patients: int = 20, seed: int = 0) -> dict:
    """1-4 studies per patient with random qualities."""
    g = torch.Generator().manual_seed(seed)
    qa = {}
    for p in range(n_patients):
        for s in range(int(torch.randint(1, 5, (1,), generator=g))):
            study_id = f"{p}_P{p}_study_{s}"
            qa[study_id] = {
                "quality": ["high", "medium", "low"][int(torch.randint(0, 3, (1,), generator=g))],
                "study_dir": f"/data/patient_{p}_P{p}/study_{s}",
                "n_time_bins": 10,
                "scan_info": {"patient_id": f"{p}_P{p}"},
            }
    return qa


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_splits_are_patient_grouped():
    splits = make_splits(fake_quality_assessment(), RATIOS, seed=42)
    for study in splits["studies"].values():
        assert study["split"] == splits["patients"][study["patient_id"]]


def test_splits_hit_ratios():
    qa = fake_quality_assessment(n_patients=100)
    splits = make_splits(qa, RATIOS, seed=42)
    n = len(splits["studies"])
    for split, ratio in RATIOS.items():
        count = sum(s["split"] == split for s in splits["studies"].values())
        assert abs(count - ratio * n) <= 4, (split, count, ratio * n)  # at most one patient (<= 4 studies) off


def test_splits_are_deterministic():
    qa = fake_quality_assessment()
    assert make_splits(qa, RATIOS, seed=42) == make_splits(qa, RATIOS, seed=42)


def test_splits_keep_existing_patients():
    qa = fake_quality_assessment(n_patients=20)
    first = make_splits({k: v for k, v in qa.items() if not k.startswith("1")}, RATIOS, seed=42)
    second = make_splits(qa, RATIOS, seed=42, existing=first)
    for patient_id, split in first["patients"].items():
        assert second["patients"][patient_id] == split
    assert set(second["studies"]) == set(qa)


def test_splits_filter_time_bins():
    qa = fake_quality_assessment()
    odd_one = next(iter(qa))
    qa[odd_one]["n_time_bins"] = 8
    splits = make_splits(qa, RATIOS, seed=42, n_time_bins=10)
    assert odd_one not in splits["studies"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_shape, size", [((10, 64, 64), (32, 48)), ((10, 20, 64, 64), (8, 32, 48))])
def test_resize_shapes(in_shape, size):
    assert ResizeTransform(size)(torch.rand(in_shape)).shape == (in_shape[0], *size)


MU_WATER, MU_AIR = 0.019, 2.3e-5


def test_hu_transforms_are_inverse():
    hu = torch.linspace(-1024, 3072, 11)
    mu = InverseHUTransform(mu_wa_mm=MU_WATER, mu_air_mm=MU_AIR)(hu)
    assert torch.allclose(HUTransform(mu_wa_mm=MU_WATER, mu_air_mm=MU_AIR)(mu), hu, atol=1e-2)


def test_inverse_hu_transform_anchors():
    """HU = 1000 (mu - mu_water) / (mu_water - mu_air): water is 0 HU and air is -1000 HU."""
    mu = InverseHUTransform(mu_wa_mm=MU_WATER, mu_air_mm=MU_AIR)(torch.tensor([0.0, -1000.0]))
    assert torch.allclose(mu, torch.tensor([MU_WATER, MU_AIR]))


def test_data_transform_range():
    transform = DataTransform(normalized_range=[-1, 1], hu_range=[-1024, 3072], mu_water_mm=MU_WATER, mu_air_mm=MU_AIR, size=[8, 16, 16])
    x = transform(torch.tensor([-1.0, 1.0]).repeat_interleave(8 * 16 * 16 // 2).reshape(1, 8, 16, 16))
    expected = InverseHUTransform(mu_wa_mm=MU_WATER, mu_air_mm=MU_AIR)(torch.tensor([-1024.0, 3072.0]))
    assert x.shape == (1, 8, 16, 16)
    assert torch.allclose(torch.stack([x.min(), x.max()]), expected)


def test_study_extent():
    scan_info = {"resampled_pixel_spacing": [0.5, 0.8], "resampled_slice_thickness": 3.0}
    assert study_extent(scan_info, (50, 100, 200)) == (150.0, 80.0, 100.0)


def test_study_id_from_dir():
    assert study_id_from_dir(Path("/x/patient_100_HM10395/study_35220329")) == "100_HM10395_study_35220329"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_poisson_noise_is_unbiased_at_high_dose():
    y = torch.full((4, 100, 32, 8), 2.0)
    noise = PoissonNoise(flux=1e12, nViews=100, nDetectorCols=32, nDetectorRows=8)
    y_noisy = noise(y)
    assert not torch.equal(y_noisy, y)
    assert torch.isclose(y_noisy.mean(), torch.tensor(2.0), atol=1e-3)


def test_collate_time_list():
    batch = [
        {"sinogram": torch.rand(10, 6, 5, 4), "volume": torch.rand(10, 3, 3, 3), "meta": {"i": i}, "study_id": str(i)}
        for i in range(2)
    ]
    collated = collate_time_list(batch)
    assert isinstance(collated["sinogram"], list) and len(collated["sinogram"]) == 10
    assert collated["sinogram"][0].shape == (2, 6, 5, 4)
    assert collated["volume"][0].shape == (2, 3, 3, 3)
    assert torch.equal(collated["sinogram"][3][1], batch[1]["sinogram"][3])
    assert collated["meta"] == [{"i": 0}, {"i": 1}]


def test_dataset_reads_pipeline_output(tmp_path):
    study_dir = tmp_path / "high" / "train" / "1_P1_study_0"
    study_dir.mkdir(parents=True)
    torch.save(torch.rand(10, 6, 5, 4), study_dir / "sinogram.pt")
    torch.save(torch.rand(10, 3, 3, 3), study_dir / "volume.pt")
    (study_dir / "meta.json").write_text(json.dumps({"study_id": "1_P1_study_0"}))
    (tmp_path / "high" / "train" / "unfinished").mkdir()  # no done marker -> ignored

    assert len(RegAndReconDataset(["high"], "train", data_root=tmp_path)) == 0
    (study_dir / "done").touch()

    loader = get_quality_loader(["high"], "train", data_root=tmp_path, load_volume=True)
    batch = next(iter(loader))
    assert isinstance(loader.dataset, Sized)
    assert len(loader.dataset) == 1
    assert batch["study_id"] == ["1_P1_study_0"]
    assert len(batch["volume"]) == 10 and batch["volume"][0].shape == (1, 3, 3, 3)


def test_raw_dataset_is_lazy(tmp_path):
    series_dir = tmp_path / "patient_1_P1" / "study_0" / "series_1_Gated,_0.0%_5"
    series_dir.mkdir(parents=True)
    torch.save(torch.rand(2, 3, 3), series_dir / "volume.pt")
    (series_dir / "scan_info.json").write_text(json.dumps({"patient_id": "1_P1"}))

    dataset = RawDataSet(tmp_path)
    assert dataset.study_dirs == [tmp_path / "patient_1_P1" / "study_0"]
    (series_dir / "volume.pt").unlink()  # nothing was loaded on construction
    with pytest.raises(FileNotFoundError):
        dataset[0]


# ---------------------------------------------------------------------------
# End-to-end pipeline on tiny synthetic data
# ---------------------------------------------------------------------------

TINY_RAY_CFG = {
    "geometry": "Parallel3dAxisGeometry", "nDetectorCols": 24, "nDetectorRows": 8, "DetectorColExtent": 60.0, "DetectorRowExtent": 20.0,
    "GantrySpeed": 1.5707963267948966, "nViews": 6, "Flux": 1e12, "rotAxis": [1.0, 0.0, 0.0], "source_radius": None, "det_radius": None,
}


@pytest.fixture
def tiny_pipeline(tmp_path, monkeypatch):
    """Raw studies of 3 patients (2 time bins of (6, 20, 20)), a quality assessment and tiny preprocessing/ray transform configs."""
    qa = {}
    for p in range(3):
        study_dir = tmp_path / "raw" / f"patient_{p}_P{p}" / "study_0"
        for t in range(2):
            series_dir = study_dir / f"series_{t}_Gated,_{t * 50}.0%_{t}"
            series_dir.mkdir(parents=True)
            torch.save(torch.rand(6, 20, 20) * 0.6 - 1.0, series_dir / "volume.pt")
            (series_dir / "scan_info.json").write_text(json.dumps({"patient_id": f"P{p}", "resampled_pixel_spacing": [2.0, 2.0], "resampled_slice_thickness": 3.0}))
        qa[f"{p}_P{p}_study_0"] = {"quality": "high", "study_dir": str(study_dir), "n_time_bins": 2, "scan_info": {"patient_id": f"P{p}"}}

    qa_path = tmp_path / "quality_assessment.json"
    qa_path.write_text(json.dumps(qa))
    preprocess_path = tmp_path / "preprocess.json"
    preprocess_path.write_text(json.dumps({
        "SHAPE": [4, 10, 10], "INITIAL_RANGE": [-1, 1], "HU_RANGE": [-1024, 3072], "mu_wa_mm": 0.019, "mu_air_mm": 2.3e-5, "time_bins": 2,
        "train_val_test_split": {"train": 0.7, "val": 0.15, "test": 0.15, "random_seed": 42},
    }))
    ray_path = tmp_path / "ray.json"
    ray_path.write_text(json.dumps(TINY_RAY_CFG))
    monkeypatch.setattr(pipeline, "PREPROCESS_CFG_PATH", preprocess_path)
    monkeypatch.setattr(pipeline, "RAY_TRAFO_CFG_PATHS", {"parallel3d": ray_path})

    return {"out_root": tmp_path / "simulated", "qa_path": qa_path, "ray_path": ray_path}


def test_pipeline_end_to_end(tiny_pipeline):
    out_root, qa_path = tiny_pipeline["out_root"], tiny_pipeline["qa_path"]
    data_pipeline(qualities=("high",), out_root=out_root, quality_assessment_path=qa_path)

    # Bookkeeping
    (run,) = [json.loads(line) for line in (out_root / "runs.jsonl").read_text().splitlines()]
    assert run["status"] == "completed" and run["kind"] == "simulation"
    assert sorted(run["results"]["simulated"]) == ["0_P0_study_0", "1_P1_study_0", "2_P2_study_0"]
    assert run["config"]["ray_trafo"] == TINY_RAY_CFG
    assert json.loads((out_root / "config.json").read_text())["config_hash"] == run["config_hash"]
    assert (out_root / "logs" / f"{run['run_id']}.log").exists()

    # Outputs
    splits = json.loads((out_root / "splits.json").read_text())
    study_dirs = sorted(out_root.glob("high/*/*"))
    assert len(study_dirs) == 3
    for study_dir in study_dirs:
        meta = json.loads((study_dir / "meta.json").read_text())
        assert study_dir.parent.name == splits["studies"][meta["study_id"]]["split"]
        assert meta["config_hash"] == run["config_hash"] and meta["run_id"] == run["run_id"]
        assert meta["shape"] == [4, 10, 10] and meta["extent"] == [18.0, 40.0, 40.0]
        assert meta["resampled_slice_thickness"] * 4 == pytest.approx(18.0)
        assert torch.load(study_dir / "volume.pt").shape == (2, 4, 10, 10)
        assert torch.load(study_dir / "sinogram.pt").shape == (2, 6, 24, 8)  # (T, views, cols, rows)

    # The quality loader reads the output
    split = splits["studies"]["0_P0_study_0"]["split"]
    batch = next(iter(get_quality_loader(["high"], split, data_root=out_root, load_volume=True, noise=PoissonNoise.from_config(TINY_RAY_CFG))))
    assert len(batch["sinogram"]) == 2 and torch.isfinite(batch["sinogram"][0]).all()


def test_pipeline_resumes_and_pins_config(tiny_pipeline):
    out_root, qa_path = tiny_pipeline["out_root"], tiny_pipeline["qa_path"]
    data_pipeline(qualities=("high",), out_root=out_root, quality_assessment_path=qa_path)
    data_pipeline(qualities=("high",), out_root=out_root, quality_assessment_path=qa_path)

    first, second = [json.loads(line) for line in (out_root / "runs.jsonl").read_text().splitlines()]
    assert len(first["results"]["simulated"]) == 3
    assert second["results"]["simulated"] == [] and len(second["results"]["skipped"]) == 3

    tiny_pipeline["ray_path"].write_text(json.dumps({**TINY_RAY_CFG, "nViews": 12}))
    with pytest.raises(ConfigMismatchError):
        data_pipeline(qualities=("high",), out_root=out_root, quality_assessment_path=qa_path)


@pytest.mark.skipif(not (DEFAULT_SIMULATED_ROOT / "splits.json").exists(), reason=f"No simulated data in {DEFAULT_SIMULATED_ROOT}")
def test_real_quality_loader():
    loader = get_quality_loader(["high", "medium"], "train", batch_size=1, load_volume=True)
    assert isinstance(loader.dataset, Sized)
    if len(loader.dataset) == 0:
        pytest.skip("No simulated high/medium training studies.")
    batch = next(iter(loader))
    meta = batch["meta"][0]
    assert len(batch["sinogram"]) == meta["n_time_bins"]
    assert batch["volume"][0].shape == (1, *meta["shape"])
    assert all(torch.isfinite(y).all() for y in batch["sinogram"])
