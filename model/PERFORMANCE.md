# HLPD training performance

Changes that speed up HLPD training and reduce its GPU memory, keeping the data size the same. Measured on 2026-10-14.

## Setup

- **GPU:** NVIDIA RTX 6000 Ada (48 GB).
- **Data:** one full-resolution study (`100_HM10395_study_93614571`): volume 50×256×256, 10 time bins, sinogram 256×256×96 per bin, `parallel3d` geometry.
- **Model:** recurrent HLPD, 3 unrolled iterations, hidden channels `[8]`, gradient checkpointing, Euler integration with 7 steps. Image domain loss.
- **Script:** `test/train_hlpd_overfit.py`:

```bash
python -m test.train_hlpd_overfit --domain image --num-iters 3 --hidden-channels 8 --steps 51
```

## Result

Mean over steps 1–50. Step 0 is excluded: it includes cuDNN autotuning and the growth of the allocator cache.

| | Before | After | Change |
|---|---|---|---|
| Time per step | 9.95 s | **3.06 s** | **3.3× faster** |
| Peak allocated GPU memory | 20.3 GB | **8.0 GB** | **−60%** |
| Peak reserved GPU memory | 25.4 GB | **8.4 GB** | **−67%** |
| 500-step overfit run | ~85 min | ~26 min | |

The runs are `20261014-104330_hlpd_overfit` (before) and `20261014-143904_hlpd_overfit` (after), in `test/runs/hlpd_overfit/`. Their timing and memory plots are in `test/plots/<run id>/`.

### Where the time goes (per step)

| Region | Before | After |
|---|---|---|
| Ray transform, forward (`hlpd/ray_trafo`) | 924 ms | 92 ms |
| Ray transform, adjoint (`hlpd/ray_trafo_adjoint`) | 929 ms | 82 ms |
| Dual net | 668 ms | 340 ms |
| Primal net | 564 ms | 168 ms |
| Motion net | 363 ms | 176 ms |
| Backward, per iteration | 1.96–2.37 s | 0.68–0.76 s |
| … of which recompute (checkpointing) | ~1.15 s | ~0.32 s |
| Loss and optimizer | ~80 ms | ~80 ms |

Before, half of each step was the ray transform. Now the conv nets dominate, especially the dual net, which runs on the sinograms (twice the size of the volume).

The memory peak used to be the backward pass of an iteration: it held the activations of all 10 time bins, about 1.5 GB each. Now the peak is in the loss (8.0 GB). The backward pass stays below 6 GB.

## Changes

The effect of each change was measured as it was added, with a single forward/backward pass of the same model (projection domain loss). Its numbers differ slightly from the overfit script's.

| # | Change | Where | Time per step | Peak allocated |
|---|---|---|---|---|
| 0 | Original | | 11.7 s | 19.5 GB |
| 1 | In-place LeakyReLU + skipping T(f=0) in iteration 1 | `model/hlpd.py` | 10.6 s | 15.4 GB |
| 2 | ASTRA directly on CUDA tensors | `operators/tomo/ray_trafo.py` | 4.7 s | 15.4 GB |
| 3 | Per-time-step checkpointing | `model/hlpd.py` | 4.6 s | 8.5 GB |
| 4 | cuDNN autotuning + expandable segments | `model/build.py` (`configure_cuda`) | 3.05 s* | 8.1 GB |

\*Measured with the image domain loss, as in the overfit script.

### 1. In-place LeakyReLU, and skipping the projection of f = 0

- **In-place LeakyReLU** (`ConvBlock`): the activation now overwrites the conv output, which the conv's backward pass does not need. This saves one hidden activation per block, about 30% of the activation memory.
- **Skipping T(f=0)** (`HLPDBlock.forward(zero_primal=...)`): with zero initialization (`zero_init=True`), the first iteration no longer forward projects the zero primal variable. This saves 10 ray transforms per forward pass, and another 10 during the recompute. With FBP initialization nothing changes.

Both changes give **bitwise identical** results.

### 2. ASTRA directly on CUDA tensors (`AstraCudaModule`)

odl's `OperatorModule` evaluates the operator on the device of its odl space, which is the CPU for our numpy-backed spaces. So each projection of a CUDA tensor went GPU → host → ASTRA's own GPU copy → host → GPU, and synchronized with the host every time.

| Per projection (50×256×256 volume) | Forward | Back |
|---|---|---|
| Through odl (before) | ~53 ms | ~55 ms |
| ASTRA kernels alone | 4.3 ms | 2.6 ms |
| odl with `impl='pytorch'` spaces | 99 ms | 44 ms |

So switching odl to GPU-backed spaces would not fix it.

`AstraCudaModule` is a drop-in replacement for `OperatorModule` that calls ASTRA directly on the CUDA tensors. It is a custom autograd function, and it reuses the projector, scaling factors and axis order of odl's own ASTRA backend. CPU tensors (e.g. in data simulation) and 2D geometries still go through odl. The FBP initialization also still goes through odl; it runs once per forward pass.

Checks:

- **Same results as odl's GPU backend.** Its projections, back projections and gradients are bitwise those of odl's GPU backend (`impl='pytorch'`). `test_astra_cuda_module_matches_odl` checks this for the parallel and cone beam geometries, together with batch axes and the CPU fallback.
- **Tiny difference from the old numpy path.** ASTRA rounds the parallel beam back projection slightly differently for device data than for host data. The new results differ from the old numpy path by about 5e-6 relative to the norm. After a full forward/backward pass of the model, the whole gradient differs by 1.3e-4 relative to its norm.
- **Same gradients as before.** They are the transpose with respect to the plain dot product, exactly as `OperatorFunction` computes them. A dot product test of the vector-Jacobian products confirms this, to within ASTRA's own forward/back projection mismatch (~1%).
- **Multiple GPUs.** ASTRA is pointed at the GPU of the input tensor, so each DDP rank uses its own GPU. Tested on cuda:0–2.
- **Synchronization.** ASTRA's calls block until its kernels are done. Before each call, torch's stream is synchronized, because ASTRA does not know about torch's streams.

### 3. Checkpointing per time step instead of per iteration

In recurrent mode, `grad_checkpointing` now checkpoints each (iteration, time bin) update instead of whole iterations (`HLPDBlock._time_step`). Everything is still recomputed exactly once, so the cost is the same. But the backward pass holds the activations of one time bin at a time instead of all ten.

The inputs of a time step are kept anyway (they are iterates or data), so this adds no memory. Results are **bitwise identical**. The non-recurrent model still checkpoints whole iterations.

The recompute regions are labeled `hlpd/iter<k>/t<t>`. `tracking/timeline.py` uses them to split the backward pass into iterations, so the timing and memory plots work as before.

### 4. cuDNN autotuning and expandable segments (`configure_cuda`)

`model.build.configure_cuda()` is called at the start of `train/train.py` and `test/train_hlpd_overfit.py`.

| Setting (final code) | Time per step | Peak reserved |
|---|---|---|
| PyTorch defaults | 3.93 s | 9.7 GB |
| + cuDNN autotuning (`cudnn.benchmark`) | **3.05 s** | 8.4 GB |
| + expandable segments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) | same | ~1.3 GB less (it removes fragmentation) |
| Exact fp32 convolutions, `configure_cuda(exact_fp32=True)` | 4.39 s | 8.4 GB |

## Numerics: TF32 convolutions

cuDNN runs float32 convolutions in TF32 by default, and the original code already did. TF32 has a 10-bit mantissa, and the large sinogram values amplify its rounding over the 30 recurrent updates. After one forward pass from a random initialization:

- f differs by **0.3%** from exact fp32.
- f differs by **1.4%** between two TF32 algorithms.

So with autotuning, results can differ between runs by about that much, depending on which algorithms cuDNN picks. With `exact_fp32=True`, autotuned results are bitwise independent of the algorithm, but each step is about 45% slower.

The default keeps TF32, as before. Use `exact_fp32=True` when you need runs to be reproducible, or to rule out TF32 when debugging. Normalizing the operator (`--normalize-operator`) should also reduce the amplification, as it brings the sinograms to the scale of the images.

## Not done

- **Not recomputing the ray transforms during backward** (selective checkpointing). This would have saved ~1.85 s per step before change 2, but only ~0.2 s now, for ~1 GB more memory.
- **bf16 autocast / `torch.compile` for the conv nets.** Measured in isolation: up to ~15% faster nets, with 30–55% less activation memory. bf16 is risky for the same reason as TF32 above, and compiling adds startup time.

The dual net is now the biggest single cost: it runs on 256×256×96 sinograms with 4→8→1 channels.

## Reproducing

```bash
python -m test.train_hlpd_overfit --domain image --num-iters 3 --hidden-channels 8 --steps 51   # timing plots in test/plots/<run id>/
python -m pytest test/test_ray_trafo.py -k astra_cuda                                            # AstraCudaModule vs odl
```
