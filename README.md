# RegAndRecon

Simultaneous Registration and Reconstruction with deep learning in 4D CT.

This repo implements a pipeline for training deep learning models that jointly reconstruct and motion-correct dynamic (time-resolved) CT volumes from simulated cone-beam projection data. The core model (HLPD) learns a per-time-bin velocity field via LDDMM-style diffeomorphic registration and uses it together with a CT ray transform to reconstruct a temporally consistent 4D volume from limited-angle, per-time-bin sinograms.

## Repo layout

| Folder | Purpose |
|---|---|
| `data/` | The data pipeline: quality triage, patient-grouped split, preprocessing, projection simulation and the quality loader (see [`data/README.md`](data/README.md)) |
| `tracking/` | Run bookkeeping: config, git commit, environment and results of each simulation (and later training) run |
| `operators/` | Physics/math operators: CT ray transform (ODL/ASTRA) and LDDMM diffeomorphic registration operators |
| `model/` | The HLPD reconstruction+registration network and its loss/quality-measure functions |
| `train/` | Training and evaluation entry points (single-GPU and multi-GPU) |
| `visualization/` | Static/dynamic (GIF) volume visualization and a Streamlit checkpoint viewer |
| `test/` | pytest suite for the operators, deformation, registration, data pipeline and run tracking |

Each of `data/`, `model/`, `train/` has matching `*.json` config files (e.g. `data/configs/preprocess.json`, `data/configs/ray_trafo/<geometry>.json`, `model/model.json`, `train/train.json`) that drive its behavior — see [Configuration](#configuration) below.

## Data pipeline (`data/`)

See [`data/README.md`](data/README.md) for the full description. In short:

1. **Triage** (`data/scripts/triage.sh`): label each raw 4D CT study as high, medium or low quality.
2. **Simulate** (`data/scripts/simulate.sh`): patient-grouped train/val/test split, preprocessing to attenuation values, and forward projection of each time bin with its own (limited-angle) geometry, either `parallel3d` (default) or `conebeam`. Clean projections are stored, and every run is tracked (config, git commit, environment) in the output directory.
3. **Load** (`data.loaders.get_quality_loader`): simulated studies filtered by quality and split, with Poisson noise added on the fly.

The `data/denoising/` and `data/segmentation/` packages are currently unused.

## Operators (`operators/`)

- **`RayTransform` / `DynamicRayTransform`** (`operators/ray_transform.py`) — CT forward/adjoint/FBP operators built on ODL with the ASTRA CUDA backend, exposed as differentiable `torch` modules (`OperatorModule`). Supports `parallel`, `cone`, and `helical` geometries. `DynamicRayTransform` builds one geometry per time bin, each covering only a partial angular range (limited-angle tomography), which better mimics real dynamic acquisitions than a single full-angle transform; it also simulates Poisson measurement noise.
- **LDDMM diffeomorphic registration** (`operators/lddmm/`):
  - `GroupAction` (`deform.py`) — applies a diffeomorphism to an image (currently geometric group action via `grid_sample`; mass-preserving action is a stated TODO), and computes the deformation's Jacobian determinant.
  - `VelocityIntegrator` (`deform.py`) — integrates a time-dependent velocity field into a flow/diffeomorphism.
  - `FlowDeformationOperator` (`deform.py`) — combines integration + group action into a single operator used by the model to deform images given a predicted velocity field.
  - `HelmholtzOperator` (`helmholtz.py`) — a Sobolev-type regularization operator (in Fourier space) on velocity fields, used for the LDDMM regularization term.
  - `LDDMMLoss` (`lddmm_loss.py`) — LDDMM-specific loss helper.

## Model (`model/`)

- **`model/blocks.py`** — reusable CNN building blocks: `ConvBlock`, `CNNModule` and residual/FiLM variants, specialized into `LambdaBlock`, `GammaBlock`, `SigmaBlock` (and residual/FiLM versions) that predict the per-iteration update terms of the model.
- **`model/model.py`** — the HLPD (Hybrid Learned Primal-Dual–style) model:
  - `HLPDModel` — base class implementing an unrolled iterative scheme (`hlpd_iterations`) that alternates between updating the sinogram-domain estimate, the image-domain estimate, and a predicted velocity field, using the `DynamicRayTransform` and `FlowDeformationOperator` operators plus learned CNN blocks. It reads geometry (volume shape, time bins, detector shape) from `data/data_pipeline_run_config.json`, `model/model.json`, and `operators/ray_trafo.json`.
  - `ResidualHLPDModel`, `HLPDModelWithLoss`, `ResidualHLPDModelWithLoss` — residual-connection and loss-fused variants.
  - `HLPDModelProfiling` — instrumented variant for timing/profiling the iterations.
  - `RecurentHLPD` — a recurrent-style variant of the model.
- **`model/loss/loss.py`** — `HLPDLoss` combines, per configurable weights (`model/loss/loss.json`):
  - image-domain data/consistency losses (`l1`, `l2`, `ssim`, `ngf`, VGG16 perceptual loss),
  - regularization on the predicted velocity field (`helmholtz` norm, spatio-temporal total variation, a "drift" penalty).
- **`model/loss/quality_measures.py`** — standalone metric implementations (MSE, SSIM, NGF, Helmholtz norm, VGG16 perceptual distance) used both in the loss and for evaluation reporting.

## Training & evaluation (`train/`)

- **`train/train.py`** — main training entry point; loads `train/train.json`, builds a `ParallelHLPDTrainer`, and runs training.
- **`train/train_parallel.py`** — `ParallelHLPDTrainer`/`HLPDTrainer`: builds the model + loss (`model/model.json`), data loaders for train/val/test splits (`RegAndReconDataset`), an Adam optimizer with `StepLR` scheduling, and handles multi-GPU training via `torch.nn.DataParallel` when more than one GPU is available. Manages timestamped checkpoint directories and periodic logging/saving (intervals configured in `train/train.json`).
- **`train/eval.py`** — loads a specific checkpoint (by run directory + epoch) and runs evaluation via the trainer's `eval()` method, reporting quality measures.

## Visualization (`visualization/`)

- **`static_visualization.py` / `dynamic_visualization.py`** — `StaticVisualization`/`DynamicVisualization` classes render axial/coronal/sagittal slices of a 4D `(B, T, D, H, W)` volume, either as a single static image or as an animated GIF over time, using metadata (pixel spacing, slice thickness) for physically correct aspect ratios. Also used interactively for the data-quality triage step (`data/triage.py`).
- **`checkpoint_viewer.py`** — a Streamlit app (`streamlit run visualization/checkpoint_viewer.py`) for browsing training runs under a checkpoint directory: loss curves and per-epoch quality-measure plots for train/val.
- **`gifs/`, `plots/`** — output directories for generated visualizations.

## Tests (`test/`)

Run `pytest` from the repository root (`pytest.ini` disables ODL's incompatible pytest plugin):
- `test_ray_trafo.py`: ray transform correctness for all geometries (shapes, adjoint consistency, FBP accuracy), including limited-angle time bins and a coupling with the deformation operator.
- `test_jacobian.py`: Jacobian determinants of the deformation operators against closed-form answers in 2D and 3D.
- `test_diffeomorphic_registration.py`: small 2D and 3D registrations on synthetic data. Run the file directly for the full demos with plots.
- `test_data_pipeline.py`: splitting, transforms, noise, loaders, and an end-to-end pipeline run on tiny synthetic data.
- `test_tracking.py`: run bookkeeping.

Tests that need real simulated data are skipped when it is not available. `test_ray_trafo_visualization.py` is a script that renders projection GIFs. Example outputs are saved under `test/plots/`.

## Configuration

Behavior is driven by JSON config files rather than CLI flags:

- `data/config.py` — raw and simulated data roots (overridable with `REGANDRECON_DATA_ROOT` / `REGANDRECON_SIMULATED_ROOT`), studies to skip, available geometries.
- `data/configs/preprocess.json` — target volume shape, HU/normalization ranges, attenuation of water and air, time bins, train/val/test split ratios.
- `data/configs/ray_trafo/<geometry>.json` — CT geometry (`parallel3d`, `conebeam`), detector shape/extent, source/detector radii, gantry speed, views and photon flux per time bin.
- `model/model.json` — channel widths for the λ/γ/σ CNN blocks, number of HLPD unrolled iterations, dropout, batch norm.
- `model/loss/loss.json` — per-term loss weights (image data/consistency terms, regularization terms).
- `train/train.json` — optimizer/scheduler settings, epoch count, logging/checkpoint intervals and directory, device, batch size.

Generated: each simulated dataset directory holds `config.json` (the exact simulation config), `splits.json` and `runs.jsonl` (one record per run). See [`data/README.md`](data/README.md#tracking-parameters). The model code still reads the removed `data/data_pipeline_run_config.json` and is due for a rewrite.

## Environment

The repo targets a CUDA GPU environment (ODL + ASTRA CUDA backend for the ray transform, PyTorch with CUDA for the model/training). A devcontainer is provided (`.devcontainer/`) based on Ubuntu 24.04; dependency installation for the segmentation submodule is pinned in `data/segmentation/requirements.txt` (torch, totalsegmentator, SimpleITK, nibabel, etc.). There is no single top-level `requirements.txt` yet — dependencies are currently implied by imports across modules (`odl`, `astra`/ASTRA toolbox, `torch`, `deepinv`, `totalsegmentator`, `SimpleITK`, `nibabel`, `cc3d`, `streamlit`, `matplotlib`, `sigpy`, `tqdm`).

## Status

This is an active research codebase. Core pieces — data pipeline, ray transform, LDDMM operators, HLPD model variants, training/eval loop, and visualization — are implemented and exercised by the scripts in `test/`. Open items called out in the code include a mass-preserving (vs. purely geometric) group action, anisotropy-aware interpolation for deformation, and further hyperparameter/architecture tuning.
