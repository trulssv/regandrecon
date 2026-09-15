# RegAndRecon

Simultaneous Registration and Reconstruction with deep learning in 4D CT.

This repo implements a pipeline for training deep learning models that jointly reconstruct and motion-correct dynamic (time-resolved) CT volumes from simulated cone-beam projection data. The core model (HLPD) learns a per-time-bin velocity field via LDDMM-style diffeomorphic registration and uses it together with a CT ray transform to reconstruct a temporally consistent 4D volume from limited-angle, per-time-bin sinograms.

## Repo layout

| Folder | Purpose |
|---|---|
| `data/` | Data loading, quality triage, preprocessing, denoising, segmentation, and the end-to-end data pipeline |
| `operators/` | Physics/math operators: CT ray transform (ODL/ASTRA) and LDDMM diffeomorphic registration operators |
| `model/` | The HLPD reconstruction+registration network and its loss/quality-measure functions |
| `train/` | Training and evaluation entry points (single-GPU and multi-GPU) |
| `visualization/` | Static/dynamic (GIF) volume visualization and a Streamlit checkpoint viewer |
| `test/` | Unit/manual test scripts for the operators, deformation and data pipeline |

Each of `data/`, `model/`, `operators/`, `train/` has a matching `*.json` config file (e.g. `data/preprocess.json`, `model/model.json`, `operators/ray_trafo.json`, `train/train.json`) that drives its behavior — see [Configuration](#configuration) below.

## Data pipeline (`data/`)

`data/data_pipeline.py` orchestrates the full per-study pipeline (`data_pipeline(steps, mode)` for `mode` in `train`/`val`/`test`), with each stage individually toggleable via a `steps` dict:

1. **Load** — `RegAndReconDataset` / `RawDataset` (`data/data_loaders.py`) load raw or processed studies (volumes, sinograms, segmentations, metadata) organized as `patient/study/series` under a data root.
2. **Quality triage & splitting** — `data/prepare_datasets.py` lets a user step through studies, visualize them, and label data quality (high/medium/low) via terminal or GUI input; results go to `data/quality_assessment.json`. It then performs a train/val/test split (ratios in `data/preprocess.json`) and organizes studies into `<quality>/<split>/` folders.
3. **Volume preprocessing** — `VolumePreprocessor` (`data/preprocess_volume.py`) stacks 3D volumes into 4D spatio-temporal tensors `(B, T, D, H, W)`, normalizes HU values to `[0, 1]`, and resizes to a target shape; it also supports rescaling back to HU / linear attenuation coefficients for simulation.
4. **Denoising** — `Denoiser` (`data/denoising/denoising.py`) applies a pretrained 2D DRUNet (via `deepinv`) slice-wise as a motion-artifact/noise cleanup step (acknowledged as an imperfect plug-and-play choice, since it's trained for 2D Gaussian noise rather than 3D Poisson CT noise).
5. **Segmentation** — `ModifiedSegmenter` (`data/segmentation/segmentation.py`) wraps `totalsegmentator` for organ/tissue segmentation, with supporting material-decomposition/tissue-composition utilities (`material_extractor.py`, `tissue.py`, `case.py`) for generating virtual non-contrast / material maps.
6. **CT simulation** — a `DynamicRayTransform` (see below) forward-projects the preprocessed volume to per-time-bin sinograms and adds Poisson noise based on an initial photon flux (`N0`) and the projected path lengths.
7. **Reconstruction baselines** — optional adjoint (backprojection) and FBP reconstructions from the simulated sinogram, for comparison against the learned model.
8. **Visualization & saving** — intermediate volumes, sinograms, segmentations and reconstructions can be saved as `.pt` tensors and visualized (axial/coronal/sagittal, static + animated) via the `visualization/` module. A run config summarizing all steps/configs used is written to `data/data_pipeline_run_config.json`.

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

- **`static_visualization.py` / `dynamic_visualization.py`** — `StaticVisualization`/`DynamicVisualization` classes render axial/coronal/sagittal slices of a 4D `(B, T, D, H, W)` volume, either as a single static image or as an animated GIF over time, using metadata (pixel spacing, slice thickness) for physically correct aspect ratios. Also used interactively for the data-quality triage step (`data/prepare_datasets.py`).
- **`checkpoint_viewer.py`** — a Streamlit app (`streamlit run visualization/checkpoint_viewer.py`) for browsing training runs under a checkpoint directory: loss curves and per-epoch quality-measure plots for train/val.
- **`gifs/`, `plots/`** — output directories for generated visualizations.

## Tests (`test/`)

Manual/exploratory test scripts (not a pytest suite in the strict sense) covering:
- `test_dynamic_ray_transform.py` / `test_dynamic_ray_transform_on_data.py` — ray transform correctness/behavior, including on real preprocessed data.
- `test_deform.py` — the LDDMM deformation operators (group action, velocity integration).
- `test_diffeomorphic_registration.py` — end-to-end diffeomorphic registration behavior.
- `test_data_pipeline.py` — the data pipeline.

Example outputs (sinograms, backprojections, deformed phantoms, registration results) are saved under `test/plots/`.

## Configuration

Behavior is driven by JSON config files rather than CLI flags:

- `data/data.json` — raw data root and organization, studies to skip.
- `data/preprocess.json` — target volume shape, HU/normalization ranges, time bins, voxel spacing, train/val/test split ratios.
- `operators/ray_trafo.json` — CT geometry (`parallel`/`cone`/`helical`), detector shape/extent, source/detector radii, photon flux `N0`, and dynamic (per-time-bin) acquisition parameters.
- `model/model.json` — channel widths for the λ/γ/σ CNN blocks, number of HLPD unrolled iterations, dropout, batch norm.
- `model/loss/loss.json` — per-term loss weights (image data/consistency terms, regularization terms).
- `train/train.json` — optimizer/scheduler settings, epoch count, logging/checkpoint intervals and directory, device, batch size.

Generated/derived config: `data/data_pipeline_run_config.json` records the exact steps and configs used for a given data-pipeline run and is read back by the model to reconstruct matching operators at train/eval time.

## Environment

The repo targets a CUDA GPU environment (ODL + ASTRA CUDA backend for the ray transform, PyTorch with CUDA for the model/training). A devcontainer is provided (`.devcontainer/`) based on Ubuntu 24.04; dependency installation for the segmentation submodule is pinned in `data/segmentation/requirements.txt` (torch, totalsegmentator, SimpleITK, nibabel, etc.). There is no single top-level `requirements.txt` yet — dependencies are currently implied by imports across modules (`odl`, `astra`/ASTRA toolbox, `torch`, `deepinv`, `totalsegmentator`, `SimpleITK`, `nibabel`, `cc3d`, `streamlit`, `matplotlib`, `sigpy`, `tqdm`).

## Status

This is an active research codebase. Core pieces — data pipeline, ray transform, LDDMM operators, HLPD model variants, training/eval loop, and visualization — are implemented and exercised by the scripts in `test/`. Open items called out in the code include a mass-preserving (vs. purely geometric) group action, anisotropy-aware interpolation for deformation, and further hyperparameter/architecture tuning.
