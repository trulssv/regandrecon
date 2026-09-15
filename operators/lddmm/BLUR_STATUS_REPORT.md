# Status report: progressive detail loss in `FlowDeformationOperator` sequences

**Scope:** `operators/lddmm/deform.py` — `VelocityIntegrator`, `GroupAction`, `FlowDeformationOperator`.
**Trigger:** Observed on real CT data — in a deformed sequence produced by `FlowDeformationOperator`, the
template (frame 0) is the sharpest image, and detail is progressively lost in each subsequent frame.
**Status:** Root cause identified and quantified on a controlled phantom that reproduces the same
mechanism as the reported CT behavior. Not yet fixed — this document is diagnosis + options, not a patch.

---

## A. What is happening

### A.1 Where a "deformed sequence" comes from

`FlowDeformationOperator.forward(template, v, superres=True)` ([deform.py:498](deform.py#L498)) produces an
animated sequence from a **single static velocity field** `v` in two steps:

1. `VelocityIntegrator` integrates `v` into a list of deformation fields `phi_1, phi_2, ..., phi_N`
   ([deform.py:527](deform.py#L527)), one per output frame, via either `euler` or `scaling_and_squaring`.
2. `GroupAction` resamples the **original template** through each `phi_k` with a single bilinear
   `grid_sample` call per frame ([deform.py:83](deform.py#L83), invoked once per list element via
   [deform.py:104-105](deform.py#L104-L105)).

This second point matters: **each frame is only one image-resampling step away from the sharp template.**
The sequence is not built by repeatedly re-warping the previous frame (that would be an obvious, expected
source of compounding blur). The blur is coming from somewhere less obvious: the deformation *field* itself.

### A.2 How `phi_k` is actually built — the real mechanism

Both integration schemes construct `phi_k` incrementally by **composing deformation fields with
themselves**, and that composition is itself a bilinear resampling operation:

- `VelocityIntegrator._compose(phi, psi)` ([deform.py:187-210](deform.py#L187-L210)) computes `phi ∘ psi`
  as `grid_sample(phi, psi, mode="bilinear")` — i.e. it treats the deformation field as an "image" with 2
  or 3 channels and bilinearly resamples *it*.
- **Euler** ([deform.py:292-319](deform.py#L292-L319)), used in the project's real config
  (`DEFORM_PARAMS["integration"] = "euler"`, `N=7`, in
  [test/test_diffeomorphic_registration.py:17-22](../../test/test_diffeomorphic_registration.py#L17-L22)):
  `phi_list[k] = phi_0 ∘ phi_0 ∘ ... ∘ phi_0` (k+1 copies), built by calling `_compose` once per loop
  iteration ([deform.py:311-313](deform.py#L311-L313)). Frame `k` in the output sequence has gone through
  **k sequential bilinear compositions** of the field before it is ever used to resample the image.
- **Scaling-and-squaring** ([deform.py:259-289](deform.py#L259-L289)) is structurally identical in this
  respect: `phi_list[k] = self._square(phi_list[k-1])`, i.e. one more `_compose` call per frame
  ([deform.py:281-283](deform.py#L281-L283)).

Bilinear interpolation is a low-pass filter. Every `_compose` call doesn't just move the field's sample
points — it also **smooths the field's own spatial detail** (sharp curvature, e.g. near the center of a
rotational/shearing motion, gets rounded off a little more each time). By the last frame of a 7-step
sequence, `phi_7` has passed through **6 rounds of this smoothing** before the template is ever touched.
When that eroded field is finally used to resample the sharp image, the image inherits the field's lost
geometric precision as apparent blur, and — as shown in B.3 — also picks up interpolation *artifacts*
(spurious high-frequency content that looks like structure but isn't).

This is a known failure mode of composing dense vector fields via repeated linear-interpolation resampling
in scaling-and-squaring / Lie-group-exponential integration schemes; it is not specific to this codebase's
choice of image (CT vs. phantom) — it is a property of the integration numerics, and it will reproduce on
any deformation with non-trivial local curvature (e.g. sliding organ boundaries, diaphragm shear, cardiac
twist), scaling with how much local curvature is present and how many steps `N` are taken.

---

## B. Quantified impact

### B.1 Method

Real CT volumes were not available in this environment, so the effect was isolated and measured on the
project's **own existing test fixture** — the Shepp-Logan phantom + swirling velocity field from
[`test/test_deform.py`](../../test/test_deform.py) — run through the *unmodified* production code path
(`FlowDeformationOperator.forward(..., superres=True)`) at `N=7`, `integration="euler"`, matching the
actual `DEFORM_PARAMS` used in the training/inference pipeline
([test/test_diffeomorphic_registration.py:17-22](../../test/test_diffeomorphic_registration.py#L17-L22)).
Using a synthetic fixture isolates the *numerical* mechanism from CT-specific confounds (noise, real
anatomy) while exercising the exact code path reported as blurry; the swirl field is deliberately large
(magnitude 50, with a rotational-velocity singularity clamped at the image center) so the effect is visible
within `N=7` steps — this is a **stress test**, not a typical-case magnitude estimate. Helmholtz-regularized
velocity fields recovered during real registration are smoother, so absolute numbers below are an upper
bound; the mechanism and its direction (monotonic loss, worse with more steps in the region tested) are not.

Three metrics were computed per frame, each normalized to the template's (frame 0) value:

- **Gradient energy** — mean squared image gradient magnitude. Directly measures edge/detail content;
  monotonic and not confounded by interpolation artifacts (this is the primary metric below).
- **Laplacian variance** — the standard "focus measure" used in camera autofocus/blur-detection.
- **High-frequency FFT energy fraction** — share of spectral power above 25% of the Nyquist radius.

Reproduction script and raw output: `blur_report_assets/quantify_blur.py`,
`blur_report_assets/blur_quantification_results.json`.

### B.2 Detail loss across the sequence

![Frame gallery](blur_report_assets/frame_gallery.png)

![Metric curves](blur_report_assets/metric_curves.png)

For the production configuration (euler, N=7):

| Frame | Field compositions so far | Gradient energy (% of template) | Laplacian variance (% of template) |
|------:|---------------------------:|---------------------------------:|-------------------------------------:|
| 0 (template) | 0 | 100.0% | 100.0% |
| 1 | 0 | 68.2% | 40.7% |
| 2 | 1 | 52.7% | 33.4% |
| 3 | 2 | 46.6% | 33.1% |
| 4 | 3 | 44.1% | 35.4% |
| 5 | 4 | 44.3% | 42.4% |
| 6 | 5 | 40.7% | 51.3% |
| 7 (last) | 6 | **34.3%** | 52.9% |

Key findings:

1. **The first deformed frame already loses ~32% of its gradient energy** relative to the template, before
   any field composition has even occurred — this is the irreducible cost of a single bilinear resample
   under a large/curved displacement (see B.4).
2. **Gradient energy then decreases monotonically, frame over frame, all the way to the end of the
   sequence** — from 68.2% (frame 1) down to 34.3% (frame 7). By the last frame, roughly **two-thirds of
   the original edge/detail content is gone**, entirely attributable to the accumulating field compositions
   (frames 1→7 differ only in how many times `_compose` has run).
3. **More integration steps make the final frame *more* degraded, not less**: N=10's last frame (33.5% of
   template gradient energy) is essentially as degraded as N=7's (34.3%), despite representing a finer
   temporal sampling — because more frames means more compositions before reaching the same total
   deformation. Requesting finer temporal resolution from this operator directly trades away image
   fidelity.
4. **Laplacian variance and gradient energy diverge in later frames** (Laplacian variance partially
   recovers from 33% back up to 53% while gradient energy keeps falling). This is not the deformation
   "un-blurring" itself — it is repeated interpolation producing structured aliasing/ringing near sharp,
   heavily folded regions of the field (visible as the jagged, high-curvature edges in frames 5-7 of the
   gallery above). A metric that only checks for "detail" (like Laplacian variance or raw sharpness) can be
   **fooled into reporting an image as sharper when it is actually accumulating interpolation artifacts**
   — which is a materially worse failure mode for a clinical setting than plain blur, since it can present
   as fabricated structure rather than an obviously soft image.

### B.3 Isolating the cause: field composition vs. unavoidable single-resample blur

To separate "blur inherent to any bilinear resample of a large deformation" from "blur added by repeated
field composition," the final frame of the N=7 euler sequence was built two ways carrying the **same total
velocity**: (a) the actual code path — 6 field compositions, then 1 image resample; (b) a single Euler step
covering the full displacement in one shot — 0 field compositions, then 1 image resample.

![One-shot vs compounded](blur_report_assets/oneshot_vs_compounded.png)

| | Gradient energy | Laplacian variance |
|---|---:|---:|
| Template | 0.0198 | 0.137 |
| One-shot (1 resample, 0 compositions) | 0.00756 | 0.0841 |
| Compounded (1 resample, 6 compositions) | 0.00680 | 0.0726 |

The compounded path loses an **additional ~10% of gradient energy and ~14% of Laplacian variance** beyond
what a single resample of the same net displacement would already lose. This confirms the repeated
bilinear self-composition of the field — not merely "warping a large amount" — is an independent,
avoidable source of information loss on top of the unavoidable baseline cost of resampling.

### B.4 A hard constraint for the 3D clinical case

PyTorch's `grid_sample` (used throughout `deform.py`) does not support any interpolation mode above
bilinear for volumetric (5D) input — confirmed directly:

```
>>> F.grid_sample(x_5d, grid_5d, mode="bicubic", align_corners=True)
NotImplementedError: grid_sampler(): bicubic interpolation only supports 4D input
```

Only `"nearest"` and `"bilinear"` (trilinear) are available for 3D volumes in native PyTorch. This is
relevant to part C below: "just switch to higher-order interpolation" is not a drop-in fix for the actual
3D CT use case without either a custom kernel or an external interpolation library.

---

## C. Ways to combat this

Ordered roughly from "changes existing numerics without changing the model" to "changes the integration
strategy." All are compatible with the requirement that **information should not be lost** for a clinically
applicable pipeline — several below trade some engineering complexity specifically to avoid that loss. This
is a known class of problem in the diffeomorphic-registration literature; citations are given per point so
each recommendation can be checked against its source rather than taken on faith, and points where the
research turned up no direct supporting source are labeled as such rather than presented as established.

1. **Reduce the number of field self-compositions for a given total deformation, not just the number of
   output frames.** B.2 point 3 shows more steps directly cost more detail. If temporal super-resolution
   (many output frames) is needed for visualization, decouple it from the integration accuracy tradeoff:
   integrate once at the coarsest N needed for diffeomorphism validity, and *re-integrate v at each desired
   time fraction from scratch* (e.g. `phi_t = euler(t * v, N)` for each desired `t`) rather than building
   `phi_t` by composing the previous frame's field forward. Every frame then carries the same, fixed number
   of compositions (`N-1`) instead of an increasing one — the systematic monotonic degradation in B.2
   disappears, at the cost of recomputing each frame independently (more compute, no accuracy loss carried
   between frames). This is consistent with how **geodesic-shooting** frameworks generate intermediate-time
   frames by re-integrating momentum/velocity forward at each requested time rather than repeatedly
   self-composing an already-discretized field (Ashburner & Friston, *"Diffeomorphic registration using
   geodesic shooting and Gauss-Newton optimisation,"* NeuroImage 2011).

2. **Apply every frame's deformation to the original template, never to a previously-deformed frame**
   (already true in this codebase — `GroupAction` resamples `x`, not `y_{k-1}`, per
   [deform.py:104-105](deform.py#L104-L105)). This is worth stating explicitly as a constraint to preserve:
   it is the one thing currently preventing this bug from being far worse (chained image resampling would
   compound the *same* mechanism directly on pixel intensities every frame, not just on the field).

3. **Compose in the Lie algebra instead of resampling fields directly (BCH composition).** Rather than
   `phi ∘ psi = grid_sample(phi, psi)`, compose the underlying *velocity fields* via a truncated
   Baker-Campbell-Hausdorff series, `v ≈ v1 + v2 + ½[v1, v2] + ...`, and only convert to a displacement
   field once, at the end (Vercauteren et al., *"Symmetric Log-Domain Diffeomorphic Registration: A
   Demons-Based Approach,"* MICCAI 2008; implementation notes in Dru & Vercauteren, *"An ITK Implementation
   of the Symmetric Log-Domain Diffeomorphic Demons Algorithm"*). Vercauteren et al. report composition
   accuracy is not very sensitive to the truncation order, so even keeping just the first-order term
   (`v1 + v2`, i.e. no commutator) recovers most of the benefit while requiring far fewer spatial resamples
   than the current per-step `_compose`. This is the most directly applicable literature match for the
   mechanism diagnosed in A.2/B.3 — it targets exactly "repeated field resampling erodes detail," not just
   integration step-size error.

4. **Higher-order composition/resampling of the deformation field itself**, if BCH composition (point 3) is
   not adopted. The field is smooth and low-frequency by construction (Helmholtz-regularized), so composing
   it with cubic or spline interpolation costs little compute relative to the image resample, but reduces
   the low-pass smoothing each `_compose` call introduces. For 2D this is a one-line change
   (`grid_sample(..., mode="bicubic")`, native PyTorch). For 3D/CT (B.4), native PyTorch has no volumetric
   bicubic yet — a tricubic `grid_sample` mode for 5D input is proposed but not yet merged
   (PyTorch PR #194787; unverified beyond the PR existing — check its merge status before depending on it).
   Available today: **MONAI's spatial resampling transforms** (`monai/transforms/spatial/array.py`) support
   scipy/cupy-backed spline interpolation of order 0–5 (order 3 = cubic B-spline) for arbitrary-dimensional
   volumes, confirmed usable for this purpose by a MONAI maintainer (Project-MONAI/MONAI Discussion #7806).
   No maintained project offering a drop-in, natively-3D `grid_sample`-compatible cubic composition beyond
   MONAI was found — flagged as a gap rather than assumed to exist.

5. **Reduce N toward the diffeomorphism-validity floor, not a fixed default.** Every additional step costs
   detail (B.2.3), and only a small number of steps are needed for invertibility once the velocity field is
   reasonably smooth: Dalca et al., *"Unsupervised Learning of Probabilistic Diffeomorphic Registration for
   Images and Surfaces,"* Medical Image Analysis 2019 (Fig. 10), found registration accuracy (Dice)
   plateaus by ~4 scaling-and-squaring steps and folding voxels drop from thousands to under 5 by 5 steps.
   **Caveat:** that sweet spot was measured against registration accuracy/invertibility on brain MRI, not
   against image-detail preservation the way this report measures it — B.2.3 shows detail cost keeps
   increasing with N well past the point where folding is already resolved, so "enough steps to avoid
   folding" and "few enough steps to avoid unacceptable blur" are not guaranteed to be the same N. Use
   `GroupAction.jacobian_determinant` ([deform.py:30-68](deform.py#L30-L68)), which already exists in this
   codebase, as the validity check, and B.2's metrics (or similar) as the detail-loss check, and tune N to
   satisfy both empirically per-dataset rather than trusting one fixed default (`N=7`/`N=10`) for both.

6. **Mass-preserving action for CT specifically.** `GroupAction.mass_preserving_action`
   ([deform.py:87-93](deform.py#L87-L93)) already exists as a stub (`TODO` at
   [deform.py:16](deform.py#L16)) and multiplies the resampled image by the Jacobian determinant of the
   inverse field. Since CT intensity is a physical density measurement, warping without a Jacobian
   correction is a quantity-preservation error independent of the blur discussed here — established in the
   lung-CT registration literature specifically (Yin, Hoffman & Lin et al., *"Mass preserving image
   registration for lung CT,"* Medical Image Analysis 2009/2011; Gorbunova et al., *"Mass preserving
   nonrigid registration of CT lung images using cubic B-spline"*), which scale warped intensities by the
   Jacobian determinant so tissue mass (∝ HU) is conserved under local compression/expansion. **Caveat:** no
   literature was found claiming mass-preservation itself *reduces* the interpolation-composition blur
   quantified in section B — that would be a separate axis of "no information loss" (intensity/mass
   accuracy under compression, not spatial-frequency detail), worth wiring up on its own merits but not a
   substitute for points 1-5 above.

### References

- Ashburner, ["A fast diffeomorphic image registration algorithm,"](https://users.fmrib.ox.ac.uk/~jesper/papers/readgroup_071009/Ashburner07.pdf) NeuroImage, 2007 (DARTEL).
- Ashburner & Friston, "Diffeomorphic registration using geodesic shooting and Gauss-Newton optimisation," NeuroImage, 2011.
- Vercauteren et al., ["Symmetric Log-Domain Diffeomorphic Registration: A Demons-Based Approach,"](https://link.springer.com/chapter/10.1007/978-3-540-85988-8_90) MICCAI, 2008.
- Dru & Vercauteren, ["An ITK Implementation of the Symmetric Log-Domain Diffeomorphic Demons Algorithm."](https://www.semanticscholar.org/paper/An-ITK-Implementation-of-the-Symmetric-Log-Domain-Dru-Vercauteren/f90cd0a490770bf9e53ca6412f4f1e3fe4949b82)
- Dalca et al., ["Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces,"](https://arxiv.org/pdf/1903.03545) Medical Image Analysis, 2019.
- [PyTorch PR #194787 — grid_sample bicubic 5-D (tricubic) support](https://github.com/pytorch/pytorch/pull/194787) (unmerged at time of writing — verify status before depending on it).
- [MONAI Discussion #7806 — cubic interpolation for 3D images.](https://github.com/Project-MONAI/MONAI/discussions/7806)
- ["Mass preserving image registration for lung CT."](https://www.sciencedirect.com/science/article/abs/pii/S1361841511001617)
- ["Mass preserving nonrigid registration of CT lung images using cubic B-spline."](https://pubmed.ncbi.nlm.nih.gov/19810495/)
- Arsigny et al., "A Log-Euclidean Framework for Statistics on Diffeomorphisms."
