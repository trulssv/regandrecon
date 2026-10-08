# data

This directory turns the raw 4D CT studies into the simulated training data for the registration and reconstruction models. It covers labelling the data quality, splitting the studies, preprocessing the volumes, simulating the projections, and loading the result filtered by quality and split.

```
raw 4D CT ──► triage ──► split ──► preprocess ──► project ──► quality loader ──► training
(DATA_ROOT)   triage.py  └────────── pipeline.py ──────────┘  loaders.py
                         (SIMULATED_ROOT/<geometry>)          (+ Poisson noise on the fly)
```

## Layout

| Path | Purpose |
|---|---|
| `config.py` | Paths (raw and simulated data roots, config files), skip list and the registry of scanner geometries. |
| `configs/preprocess.json` | Target shape, intensity ranges, attenuation of water and air, number of time bins, split ratios and seed. |
| `configs/ray_trafo/<geometry>.json` | Ray transform parameters per geometry: `parallel3d` (default) and `conebeam`. |
| `quality_assessment.json` | Quality labels from the triage. Written by `triage.py`, read by `pipeline.py`. |
| `triage.py` | Manual quality assessment: shows each raw study and asks for a high, medium or low label. |
| `triage.ipynb` | Notebook version of the triage, for when no display is available (e.g. VS Code over SSH). Takes the same `resume` option as `triage.py`. |
| `pipeline.py` | The data pipeline: patient-grouped split, preprocessing and projection (`data_pipeline`, `make_splits`). |
| `transforms.py` | Volume transforms (resize, normalized range → HU → attenuation in mm⁻¹) and the Poisson noise model for projections. |
| `loaders.py` | Raw loader (`RawDataSet`, `load_study`) and quality loader (`RegAndReconDataset`, `get_quality_loader`). |
| `utils.py` | JSON helpers. |
| `scripts/` | `triage.sh` and `simulate.sh`. Both activate the `regandrecon` conda env and run from the repository root. |
| `denoising/`, `segmentation/` | **Currently unused.** Left over from an earlier version of the pipeline and kept for possible later use. |

## Running the pipeline

```bash
data/scripts/triage.sh                    # 1. label every raw study (overwrites quality_assessment.json); --resume to continue a previous triage, --no-gui-input for terminal input
data/scripts/simulate.sh                  # 2. split, preprocess and simulate the high-quality studies with the parallel3d geometry
data/scripts/simulate.sh --qualities high medium --geometry conebeam
```

`simulate.sh` can be interrupted and restarted at any time. Studies that are already done are skipped unless you pass `--overwrite`.

Then load the data for training:

```python
from data.loaders import get_quality_loader
from data.transforms import PoissonNoise
from data.utils import load_json
from data.config import RAY_TRAFO_CFG_PATHS

noise = PoissonNoise.from_config(load_json(RAY_TRAFO_CFG_PATHS["parallel3d"]))  # pass flux=... to change the dose
loader = get_quality_loader(["high"], "train", batch_size=1, load_volume=True, noise=noise)
batch = next(iter(loader))
batch["sinogram"]  # list over time of (B, views, cols, rows) tensors, with a new noise realization in every epoch
batch["volume"]    # list over time of (B, D, H, W) attenuation values in mm^-1
batch["meta"]      # list of B dicts: extent, voxel spacing, time steps, study id, config hash, ...
```

## Pipeline stages

1. **Triage** (`triage.py`): labels each raw study as high, medium or low quality. Entries are keyed by study id (`<patient_id>_study_<n>`) and also store the study directory, the number of time bins and the scan info.
2. **Split** (`pipeline.make_splits`): train/val/test split grouped by patient, so all studies of a patient end up in the same split and no anatomy is shared between splits. Ratios and seed come from `configs/preprocess.json`. When more studies are labelled, patients that already have a split keep it, and only new patients are assigned. Studies whose number of time bins doesn't match `time_bins` are left out.
3. **Preprocess** (`transforms.DataTransform`): resizes the `(T, D, H, W)` stack to `SHAPE` and maps it from the normalized range to HU and then to attenuation values in mm⁻¹, using HU = 1000 (μ − μ_water) / (μ_water − μ_air). The bottom of the HU range lies below air, so attenuation values can be slightly negative. This is intended.
4. **Project** (`pipeline.simulate_study`): forward projects each time bin with its own `RayTransform` geometry. Each bin lasts one time unit, because no gating times are available, so a bin covers `GantrySpeed` rad (π/2 by default, limited angle) with `nViews` views. Only clean line integrals are stored. The noise is added by the loader.
5. **Load** (`loaders.get_quality_loader`): yields the model-side format, where time is a list and batch and space are tensors.

## On-disk layout

```
DATA_ROOT/patient_<id>/study_<n>/series_<...>_Gated,_<pct>%_<k>/{volume.pt, scan_info.json}   raw input

SIMULATED_ROOT/<geometry>/
    config.json          the simulation config (preprocessing, ray transform, time step) and its hash
    splits.json          patient → split, and per study: split, quality, patient id, raw study dir
    runs.jsonl           one record per pipeline run (see "Tracking parameters" below)
    logs/<run_id>.log    log output per run
    <quality>/<split>/<study_id>/
        volume.pt        (T, D, H, W) attenuation values in mm^-1
        sinogram.pt      (T, views, cols, rows) clean line integrals
        meta.json        scan info, extent, voxel spacing of the resized grid, time steps, config hash, run id
        done             marker written last. The loader only includes finished studies.
```

`DATA_ROOT` and `SIMULATED_ROOT` default to `/media/truls-svensson/LDDMM/{processed,simulated}`. They can be overridden with the `REGANDRECON_DATA_ROOT` and `REGANDRECON_SIMULATED_ROOT` environment variables, for example to run the pipeline on test data.

## Tracking parameters

Every pipeline run is tracked with `tracking.start_run` (see `tracking/run.py`), which records:

- **config**: everything that determines the simulated data (geometry, preprocessing and ray transform parameters, time step), plus its hash.
- **git**: commit, branch, and whether there were uncommitted changes.
- **environment**: Python, platform, host, GPU, and versions of torch, odl, astra and numpy.
- **args**: how the run was executed (qualities, output directory, overwrite).
- **results**: which studies were simulated or skipped, and hashes of the quality assessment and split that were used.

`config.json` ties a `SIMULATED_ROOT/<geometry>` directory to one config. If any parameter changes, the next run fails with `ConfigMismatchError` rather than mixing data from different configs. To keep both datasets, simulate into another `--out-root`. Each study's `meta.json` names the config hash and run that produced it.

The same tooling is meant for training: `start_run("training", config={model, optimizer, data: {simulation config hash, qualities, flux, ...}}, log_dir=run_dir)`. It gives each training run the same record, plus per-step metrics through `run.log_metrics(step, loss=...)`. Note that the noise level (flux) is chosen in the loader, so it is a training parameter and not a simulation parameter.

## Conventions and notes

- Units: volumes are attenuation values in mm⁻¹, and extents are in mm, ordered (D, H, W). The rotation axis is D (`rotAxis = [1, 0, 0]`).
- Time: stored and preprocessed as a leading tensor dimension. On the model side (the deformation and ray transform operators), time is a `list[Tensor]` of length T.
- NOTE: each study carries its own physical extent, and with it its own ray transform. All current studies share the same extent, so one ray transform serves a whole batch. TODO: check or enforce equal extents within a batch if studies with other extents are added.

## Tests

`pytest` (configured in `pytest.ini` at the repository root) covers this directory in `test/test_data_pipeline.py`, including an end-to-end pipeline run on tiny synthetic data, and the run tracking in `test/test_tracking.py`.
