# RegAndRecon

Simultaneous Registration and Reconstruction with deep learning in 4D CT.

This repo implements a pipeline for training deep learning models that jointly reconstruct and motion-correct dynamic (time-resolved) CT volumes from simulated projection data. Each time bin of a 4D CT study is projected with its own limited-angle geometry (a parallel beam by default, or a cone beam), so that no single time bin has complete data. The core model (HLPD, a hybrid learned primal-dual scheme) reconstructs every time bin and estimates the cyclic (respiratory) motion between consecutive time bins as stationary velocity fields. The motion moves each reconstruction to the next time bins, where their data cover the angles it misses.

## Repo layout

| Folder | Purpose |
|---|---|
| `data/` | The data pipeline: quality triage, patient-grouped split, preprocessing, projection simulation and the quality loader (see [`data/README.md`](data/README.md)) |
| `operators/` | Physics/math operators: the CT ray transform (ODL/ASTRA) and the diffeomorphic deformation operators |
| `model/` | The HLPD model, its loss, and the config and builder shared by the training scripts (training speed and memory: [`model/PERFORMANCE.md`](model/PERFORMANCE.md)) |
| `train/` | The training entry point (one or several GPUs) and its config |
| `tracking/` | Run bookkeeping (config, git commit, environment and results of each simulation and training run), profiling, and the live training plots |
| `visualization/` | Static/dynamic (GIF) volume visualization and a Streamlit checkpoint viewer |
| `research/` | Investigations of the deformation operators (blur, boundary flux), with their scripts and reports |
| `test/` | pytest suite, benchmarks, and the single-study overfit script |

The data pipeline and the training are driven by JSON config files (e.g. `data/configs/preprocess.json`, `data/configs/ray_trafo/<geometry>.json`, `train/configs/hlpd.json`), see [Configuration](#configuration) below.

## Data pipeline (`data/`)

See [`data/README.md`](data/README.md) for the full description. In short:

1. **Triage** (`data/scripts/triage.sh`): label each raw 4D CT study as high, medium or low quality.
2. **Simulate** (`data/scripts/simulate.sh`): patient-grouped train/val/test split, preprocessing to attenuation values, and forward projection of each time bin with its own (limited-angle) geometry, either `parallel3d` (default) or `conebeam`. Clean projections are stored, and every run is tracked (config, git commit, environment) in the output directory.
3. **Load** (`data.loaders.get_quality_loader`): simulated studies filtered by quality and split, with Poisson noise added on the fly.

The `data/denoising/` and `data/segmentation/` packages are currently unused.

## Operators (`operators/`)

Conventions: volumes are `(B, D, H, W)` with D the slice axis, which is also the rotation axis of the scanner (`rotAxis = (1, 0, 0)`). Projection data are `(B, nViews, nDetectorCols, nDetectorRows)`. Images are attenuation values in mm^-1 and lengths are in mm, from the physical extent of each study.

### Ray transform (`operators/tomo/ray_trafo.py`)

- **`RayTransform`** — CT forward and back projection and FBP on ODL with the ASTRA CUDA backend, as differentiable torch operators. Geometries: `Parallel2dGeometry`, `FanBeamGeometry` (2D), `Parallel3dAxisGeometry`, `ConeBeamGeometry` (3D, with optional detector curvature and helical pitch).
  - **Time bins**: `_init_ray_transform(shape, extent, time_steps)` builds one geometry per time bin. Time bin t continues the gantry rotation where t-1 stopped and covers an arc of `GantrySpeed * time_steps[t]` with `nViews * time_steps[t]` views, so each time bin is a limited-angle problem (90 degrees per time bin in the simulated data). `HLPDModel.set_geometry` calls it for every study. The operators of the 8 most recently used geometries are cached, so switching between studies that share a geometry costs nothing.
  - `__call__` / `adjoint` / `FBP` take a tensor (static geometry) or a list with one tensor per time bin; `forward_bin`, `adjoint_bin` and `fbp_bin` apply the operator of one time bin. The FBP (Ram-Lak) is built on first use, as it takes about 0.4 GB of host memory per time bin.
  - **Coverage**: `bin_arcs()` is the arc of each time bin, `required_arc()` the arc needed for complete data (pi for a parallel beam, pi plus the fan angle for a cone beam), and `bins_for_complete_data()` the number of consecutive time bins that together give complete data (2 for 90 degree time bins of a parallel beam, 4 for 45 degrees).
  - `add_poisson_noise` simulates noisy measurements (Lambert-Beer, Poisson counts) from clean projections.
- **`AstraCudaModule`** — runs ASTRA directly on CUDA tensors for 3D geometries, with exactly the projections and gradients of ODL's `OperatorModule`, which copies every call through host memory (about 10x slower). Used automatically where it applies.
- `operator_norm` (power iteration, e.g. for ||T_t||), `photon_count`, and the module-level `add_poisson_noise`.

### Deformation (`operators/lddmm/`)

Conventions: a velocity field is `(B, D, H, W, 3)` in mm, with the channels in `grid_sample`'s (x, y, z) = (W, H, D) order. The deformation of an image f by a stationary velocity field v is W(f, v) = f o exp(v)^-1, where exp(v) is the flow of v over unit time.

- **`resample`** (`deform.py`) — the interpolation shared by the operators below, with two backends:
  - `grid_sample`: torch's (tri)linear interpolation. Fast, but smooths the image at every resampling.
  - `grid_pull`: B-spline interpolation from torch-interpol, of a given `order` (an int, or one per axis (D, H, W), e.g. `[3, 1, 1]` for cubic along the coarse slice axis only). The image is prefiltered, so it is interpolated rather than smoothed, which removes most of the blur at a higher cost (see `test/test_warp_benchmark.py`). Its double backward is not supported.
- **`GroupAction`** (`deform.py`) — applies a deformation (a sampling grid) to an image: the `geometric` action f o phi^-1, or the `mass_preserving` action, which also multiplies with the Jacobian determinant (`jacobian_determinant`).
- **`VelocityIntegrator`** (`deform.py`) — integrates a stationary velocity field into a deformation by `scaling_and_squaring` (N squarings of v / 2^N) or `euler` (N steps), optionally returning the intermediate deformations (`superres`). `_compose` composes two deformations. With the `grid_pull` backend it resamples the displacement rather than the deformation, as a spline with a boundary condition does not reproduce the identity near the edges.
- **`FlowDeformationOperator`** (`deform.py`) — W(f, v): normalizes v (mm) to grid_sample's [-1, 1] with the physical extent, integrates it and applies the group action. A list of velocity fields gives a list of deformed images, each from the same f. `backend` sets the interpolation of the image, `integrator_backend` that of the compositions, and `order` the spline order of `grid_pull`.
- **`HelmholtzOperator`** (`helmholtz.py`) — L = (alpha - gamma Laplacian)^beta, applied in the Fourier domain (periodic boundaries) with the frequencies in cycles/mm, for the Sobolev regularization ||L v||^2 of the velocity fields.
- **`LDDMMloss`** (`lddmm_loss.py`) — the registration loss of a single template deformed by each velocity field, against target images or (with a ray transform) their data. Used by `test/test_diffeomorphic_registration.py`; the HLPD model uses `model/hlpd_loss.py` instead.

## Model (`model/`)

### HLPD model (`model/hlpd.py`)

`HLPDModel` maps the data g (one sinogram per time bin) to three lists over the T time bins:

- **f_t**: the reconstruction of time bin t, in mm^-1,
- **v_t**: the stationary velocity field from time bin t to the next one, t+1 mod T, `(B, D, H, W, 3)` in mm. The motion is cyclic: v_{T-1} maps the last time bin back to the first,
- **w_t** = W(f_{t-1}, v_{t-1}): the prediction of time bin t by the motion from the previous time bin, with w_0 = W(f_{T-1}, v_{T-1}).

Call `set_geometry(model, loss, meta)` (`model/build.py`) for each study before the forward pass: it sets the physical extent of the study in the ray transform, the deformation operator and the loss's Helmholtz operator, builds the time bins' ray transforms, and estimates the operator norms ||T_t|| (once per geometry, see `HLPDModel.set_geometry`).

**Iterations.** `HLPDUnrolled` applies `num_iters` iterations of an `HLPDBlock` to the state (f, h, v, w), where h is the dual variable (one sinogram per time bin). The state is initialized with zeros, or (`zero_init = false`) with the FBP of each time bin (the FBP stored by the data pipeline) and the data, with w = f and v = 0. Each update is residual on the same time bin's iterate of the previous iteration. For time bin t, an iteration of the recurrent block:

| Step | Update | Inputs |
|---|---|---|
| dual | h_t += Gamma(...) | h_t (previous iteration), h_{t-1}, T_t f_t (f_t of the previous iteration), g_t |
| primal | f_t += Lambda(...) | f_t (previous iteration), f_{t-1}, T_t^* h_t (the updated h_t), w_t |
| motion | v_t += Sigma(...) | v_t (previous iteration), v_{t-1}, the updated f_t, f_{t+1} (the image v_t should map f_t to), w_{t+1} (its previous prediction) |
| warp | w_{t+1} = W(f_t, v_t) | the updated f_t and v_t |

The nets Gamma, Lambda and Sigma are `ConvStack`s: 3D conv layers with GroupNorm (`batch_norm`) and LeakyReLU, whose final layer is zero initialized (`zero_init_final`), so that an untrained model returns its initialization.

- **Recurrent** (`recurrent = true`): the time bins are updated one after the other within an iteration, so that time bin t sees the iterates (f, h, v) of time bin t-1 of the current iteration and the fresh prediction w_t; time bin 0 sees those of the last time bin from the previous iteration. The nets are separate per time bin, or shared across time bins (`HLPDBlockParams.shared`).
- **Non-recurrent**: all time bins are stacked as channels of one set of nets (very memory hungry at full resolution), without the inputs from the previous time bin, and w is computed after the motion update.
- **Sharing and checkpointing**: the block is separate per iteration or shared (`HLPDUnrolledParams.shared`). With `grad_checkpointing`, the recurrent block checkpoints each (iteration, time bin) update, so that the backward pass holds the activations of one time bin at a time.

`net_channels` gives the in and out channels of the three nets in both modes.

**Normalization.** The iterations work in their own units (`NormalizationParams`), so that the nets see O(1) values:

- `intensity = "water"`: images in units of the attenuation of water (c = 1 / mu_water), instead of mm^-1 (`"physical"`),
- `operator = true`: the ray transform of time bin t scaled by s_t = 1 / ||T_t|| (`NormalizedRayTransform`), and the data accordingly.

`IterationUnits` (`HLPDModel.units`) converts images (c x) and data (s_t c g_t) to these units and back, and is shared with the loss, so that the loss uses the same normalization as the iterations. The outputs of the model are physical (mm^-1 and mm).

**Parameters** (dataclasses in `model/hlpd.py`): `HLPDUnrolledParams` (iterations, sharing, checkpointing), `HLPDBlockParams` (recurrent, sharing across time bins), `ConvStackParams` (hidden channels, which must be divisible by 4 for the GroupNorm, kernel size, dropout, ...), `NormalizationParams`, `DeformParams` (integration steps N, `action`, `integration`, `backend`, `integrator_backend`, `order`, see the deformation operators above) and `RayTrafoParams` (the geometry, which must be the one the data was simulated with, see `simulation_ray_trafo_params`).

### Loss (`model/hlpd_loss.py`)

`HLPDLoss` is a joint reconstruction and registration loss of the outputs (w, f, v), with P_t = T_t and y_t = g_t in the projection domain (unsupervised), or P_t = id and y_t = x_t, the ground truth volumes, in the image domain (supervised):

| Term | Weight | Definition (summed over the time bins t, cyclically) |
|---|---|---|
| data | 1 | \|\|P_t f_t - y_t\|\|^2 |
| coupling | `lambda_coupling` | \|\|w_t - f_t\|\|^2 |
| registered data | `lambda_registered` | 1/J sum_{j=1..J} \|\|P_t W(f_{t-j}, phi_{t-j -> t}) - y_t\|\|^2 |
| window (optional) | `lambda_window` | 1/(2K+1) sum_{i=-K..K} \|\|P_{t+i} fbar_t - y_{t+i}\|\|^2, fbar_t the mean of f_{t-K}, ..., f_{t+K} |
| regularization | `lambda_reg` | \|\|L v_t\|\|^2, L the Helmholtz operator |

The norms are means over the voxels or detector pixels, and all terms but the regularization are computed in the units of the iterations.

- The data term of a time bin is blind to the angles it misses (the missing wedge of its limited arc), so the reconstructions keep the limited-angle artifacts. The **registered data term** moves each f_s with the motion to the next J time bins and checks it against their data, which together with its own cover the complete arc. J defaults to the number the geometry needs (`RayTransform.bins_for_complete_data() - 1`: 1 for 90 degree time bins, 3 for 45 degrees), or is set with `registered_gaps`. The term with j = 1 is \|\|P_t w_t - y_t\|\|^2; for j > 1 the motions v_{t-j}, ..., v_{t-1} are composed and f_{t-j} is warped once. It is minimized by the true reconstructions with the true motion, so it also rewards the motion.
- The **coupling** term ties each reconstruction to its prediction by the motion. The data term must be kept: without it, f_t could follow w_t for any v.
- The **window** term (off by default) checks the mean of 2K+1 time bins against their data. It is independent of the motion and biased towards no motion, and is meant as a warm-up at most.

### Config and builder (`model/build.py`)

- `HLPDConfig` holds the model and loss parameters (the `model` section of the training config), with `LossParams` for the loss (`domain`, the weights above, `registered_gaps`, `window_k`, and the Helmholtz `(alpha, gamma, beta)`).
- `build_hlpd(config, ray_trafo_params, T, shape, device, mu_water)` builds the model and its loss, and `set_geometry(model, loss, meta)` sets a study's geometry in both.
- `simulation_ray_trafo_params(data_root)` and `simulation_mu_water(data_root)` read the ray transform and the attenuation of water the data under `data_root` was simulated with.
- `configure_cuda` sets the allocator and cuDNN options of the training scripts (see `model/PERFORMANCE.md`).

`model/PERFORMANCE.md` documents the speed and memory optimizations of the training, measured on 2026-10-14 with the earlier, template-based version of the model.

## Training (`train/`)

- **`train/train.py`** — trains the HLPD model on the simulated studies, on one or several GPUs (DistributedDataParallel, one study per GPU and step, gradients averaged over the GPUs). It validates every `val_every` epochs, and writes checkpoints (`last.pt`, `best.pt` by validation loss, and every `checkpoint_every` epochs) with the model, optimizer, scheduler and RNG states, which `--resume` continues from. The per-step metrics (the loss terms, and the image domain errors of the reconstructions, `image_mse`, and of their predictions, `warp_mse`) and the run record go to `--log-dir`, and the live plots (losses, GPU memory, step times, and a prediction GIF of a validation study) to `--plot-dir/<run_id>/`.

  ```bash
  torchrun --standalone --nproc_per_node=4 -m train.train --config train/configs/hlpd.json
  python -m train.train --config train/configs/hlpd.json --max-steps 5      # single GPU smoke test
  ```

- **`test/train_hlpd_overfit.py`** — overfits the model to a single study, as a check that it can converge at all, with the model and loss parameters as command line options and optional profiling (`--profile`, the time per module).

  ```bash
  python -m test.train_hlpd_overfit --domain projection --steps 500 --profile
  ```

Checkpoints are only compatible with the architecture they were trained with: the input channels of the nets depend on the model version (see `net_channels`).

## Visualization (`visualization/`)

- **`static_visualization.py` / `dynamic_visualization.py`** — `StaticVisualization`/`DynamicVisualization` classes render axial/coronal/sagittal slices of a 4D `(B, T, D, H, W)` volume, either as a single static image or as an animated GIF over time, using metadata (pixel spacing, slice thickness) for physically correct aspect ratios. Also used interactively for the data-quality triage step (`data/triage.py`).
- **`checkpoint_viewer.py`** — a Streamlit app (`streamlit run visualization/checkpoint_viewer.py`) for browsing training runs under a checkpoint directory: loss curves and per-epoch quality-measure plots for train/val.
- **`gifs/`, `plots/`** — output directories for generated visualizations.

## Tests (`test/`)

Run `pytest` from the repository root (`pytest.ini` disables ODL's incompatible pytest plugin):
- `test_ray_trafo.py`: ray transform correctness for all geometries (shapes, adjoint consistency, FBP accuracy), including limited-angle time bins, the number of time bins for complete data, and a coupling with the deformation operator.
- `test_hlpd.py`: the HLPD model and its loss against a mock ray transform: shapes, gradients, checkpointing, the cyclic predictions w, the normalization, and every loss term.
- `test_deform_backends.py`: the `grid_sample` and `grid_pull` interpolation backends of the deformation operators.
- `test_velocity_integration.py`: both integration schemes integrate the full velocity field.
- `test_jacobian.py`: Jacobian determinants of the deformation operators against closed-form answers in 2D and 3D.
- `test_helmholtz.py`: the Helmholtz operator on plane waves, in 2D and 3D.
- `test_diffeomorphic_registration.py`: small 2D and 3D registrations on synthetic data. Run the file directly for the full demos with plots.
- `test_data_pipeline.py`: splitting, transforms, noise, loaders, and an end-to-end pipeline run on tiny synthetic data.
- `test_tracking.py`: run bookkeeping.
- `test_warp_benchmark.py`: speed and quality of the interpolation backends on a real volume (also a script: `python test/test_warp_benchmark.py`).

Tests that need real simulated data are skipped when it is not available. `test_ray_trafo_visualization.py` is a script that renders projection GIFs. Example outputs are saved under `test/plots/`.

## Configuration

- `data/config.py` — raw and simulated data roots (overridable with `REGANDRECON_DATA_ROOT` / `REGANDRECON_SIMULATED_ROOT`), studies to skip, available geometries.
- `data/configs/preprocess.json` — target volume shape, HU/normalization ranges, attenuation of water and air, time bins, train/val/test split ratios.
- `data/configs/ray_trafo/<geometry>.json` — CT geometry (`parallel3d`, `conebeam`), detector shape/extent, source/detector radii, gantry speed, views and photon flux per time bin.
- `train/configs/hlpd.json` — the training config:
  - `model`: the `HLPDConfig`, with the sections `unrolled`, `block`, `conv_stack`, `deform`, `normalization` and `loss` (the dataclasses in `model/hlpd.py` and `LossParams` in `model/build.py`), and `zero_init`. The ray transform is not part of it: it is read from the simulated data (`simulation_ray_trafo_params`).
  - `data`: the qualities of the training and validation studies.
  - `optimizer` and `scheduler`: a `torch.optim` optimizer and (optional) `torch.optim.lr_scheduler` scheduler by name, with their parameters.
  - `training`: epochs, gradient clipping, validation and checkpoint intervals, and the seed.

Generated: each simulated dataset directory holds `config.json` (the exact simulation config), `splits.json` and `runs.jsonl` (one record per run). See [`data/README.md`](data/README.md#tracking-parameters). Training runs are recorded in the same way under `--log-dir` (see `tracking/run.py`).

## Environment

The repo targets a CUDA GPU environment (ODL + ASTRA CUDA backend for the ray transform, PyTorch with CUDA for the model/training). A devcontainer is provided (`.devcontainer/`) based on Ubuntu 24.04; dependency installation for the segmentation submodule is pinned in `data/segmentation/requirements.txt` (torch, totalsegmentator, SimpleITK, nibabel, etc.). There is no single top-level `requirements.txt` yet — dependencies are currently implied by imports across modules (`odl`, `astra`/ASTRA toolbox, `torch`, `deepinv`, `totalsegmentator`, `SimpleITK`, `nibabel`, `cc3d`, `streamlit`, `matplotlib`, `sigpy`, `tqdm`).

## Status

This is an active research codebase. The data pipeline, the ray transform, the deformation operators, the HLPD model with its loss, and the training loop are implemented and tested. Open problems include the velocity fields staying close to zero early in training, while the limited-angle artifacts dominate the reconstructions, and the small receptive field of the conv nets for the motion estimation.
