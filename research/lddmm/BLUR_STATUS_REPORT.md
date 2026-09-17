# Status report: progressive detail loss in `FlowDeformationOperator` sequences

**Scope:** `operators/lddmm/deform.py` — `VelocityIntegrator`, `GroupAction`, `FlowDeformationOperator`.
**Trigger:** Observed on real CT data — in a deformed sequence produced by `FlowDeformationOperator`, the
template (frame 0) is the sharpest image, and detail is progressively lost in each subsequent frame.
**Status:** Root cause identified and quantified on a controlled phantom that reproduces the same
mechanism as the reported CT behavior. Not yet fixed — this document is diagnosis + options, not a patch.

---

## A. What is happening

### A.1 Where a "deformed sequence" comes from

`FlowDeformationOperator.forward(template, v, superres=True)` ([deform.py:498](../../operators/lddmm/deform.py#L498)) produces an
animated sequence from a **single static velocity field** `v` in two steps:

1. `VelocityIntegrator` integrates `v` into a list of deformation fields `phi_1, phi_2, ..., phi_N`
   ([deform.py:527](../../operators/lddmm/deform.py#L527)), one per output frame, via either `euler` or `scaling_and_squaring`.
2. `GroupAction` resamples the **original template** through each `phi_k` with a single bilinear
   `grid_sample` call per frame ([deform.py:83](../../operators/lddmm/deform.py#L83), invoked once per list element via
   [deform.py:104-105](../../operators/lddmm/deform.py#L104-L105)).

This second point matters: **each frame is only one image-resampling step away from the sharp template.**
The sequence is not built by repeatedly re-warping the previous frame (that would be an obvious, expected
source of compounding blur). The blur is coming from somewhere less obvious: the deformation *field* itself.

### A.2 How `phi_k` is actually built — the real mechanism

Both integration schemes construct `phi_k` incrementally by **composing deformation fields with
themselves**, and that composition is itself a bilinear resampling operation:

- `VelocityIntegrator._compose(phi, psi)` ([deform.py:187-210](../../operators/lddmm/deform.py#L187-L210)) computes `phi ∘ psi`
  as `grid_sample(phi, psi, mode="bilinear")` — i.e. it treats the deformation field as an "image" with 2
  or 3 channels and bilinearly resamples *it*.
- **Euler** ([deform.py:292-319](../../operators/lddmm/deform.py#L292-L319)), used in the project's real config
  (`DEFORM_PARAMS["integration"] = "euler"`, `N=7`, in
  [test/test_diffeomorphic_registration.py:17-22](../../test/test_diffeomorphic_registration.py#L17-L22)):
  `phi_list[k] = phi_0 ∘ phi_0 ∘ ... ∘ phi_0` (k+1 copies), built by calling `_compose` once per loop
  iteration ([deform.py:311-313](../../operators/lddmm/deform.py#L311-L313)). Frame `k` in the output sequence has gone through
  **k sequential bilinear compositions** of the field before it is ever used to resample the image.
- **Scaling-and-squaring** ([deform.py:259-289](../../operators/lddmm/deform.py#L259-L289)) is structurally identical in this
  respect: `phi_list[k] = self._square(phi_list[k-1])`, i.e. one more `_compose` call per frame
  ([deform.py:281-283](../../operators/lddmm/deform.py#L281-L283)).

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
   [deform.py:104-105](../../operators/lddmm/deform.py#L104-L105)). This is worth stating explicitly as a constraint to preserve:
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
   `GroupAction.jacobian_determinant` ([deform.py:30-68](../../operators/lddmm/deform.py#L30-L68)), which already exists in this
   codebase, as the validity check, and B.2's metrics (or similar) as the detail-loss check, and tune N to
   satisfy both empirically per-dataset rather than trusting one fixed default (`N=7`/`N=10`) for both.

6. **Mass-preserving action for CT specifically.** `GroupAction.mass_preserving_action`
   ([deform.py:87-93](../../operators/lddmm/deform.py#L87-L93)) already exists as a stub (`TODO` at
   [deform.py:16](../../operators/lddmm/deform.py#L16)) and multiplies the resampled image by the Jacobian determinant of the
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

---

## D. Empirical test: does `mass_preserving_action` help, hurt, or leave the composition blur alone?

Point C.6 flagged mass-preserving as worth wiring up for CT intensity/quantity conservation but
explicitly noted no literature was found claiming it *also* reduces the composition blur quantified
in section B, leaving that as an open question. This section tests it directly, on the same fixture,
rather than leaving it as speculation — and in the process turns up a calibration bug in
`jacobian_determinant` itself that turned out to dominate the result.

### D.1 Method

`GroupAction.mass_preserving_action` ([deform.py:87-93](../../operators/lddmm/deform.py#L87-L93)) and `geometric_action`
([deform.py:70-85](../../operators/lddmm/deform.py#L70-L85)) were applied to the **same** `phi_k` sequence (Shepp-Logan
phantom, swirl velocity field, `N=7`, `integration="euler"`, matching `DEFORM_PARAMS`, per B.1) so any
difference is attributable only to the action, not to a re-integrated/slightly different field.
Applying both actions to an identical `phi_k` — rather than running `FlowDeformationOperator.forward`
twice with `action="geometric"` vs `action="mass_preserving"` — removes any doubt about the two
sequences seeing the same field.

Before trusting anything derived from `jacobian_determinant`, it was sanity-checked directly against
hand-constructed fields with a known-by-construction determinant sign/value (identity, uniform
isotropic scaling, a single-axis mirror flip) — this is what surfaced the calibration issue in D.2.
Three checks were run per the task brief: the three B.2 sharpness metrics for both actions per frame,
total/mean image intensity per frame relative to the template for both actions, and whether/where
`jacobian_determinant` goes negative (see D.2 for why "negative" is not simply "folded" in this
codebase). A supplementary low-magnitude run (`magnitude=5` instead of the stress-test `magnitude=50`
used elsewhere in this report) was added because the primary fixture turned out — see D.4 — to violate
the "closed boundary" assumption under which mass-preservation is meant to apply; this was not assumed
going in, it was checked. `grep`-ing the repository found no existing test exercising
`jacobian_determinant` or `mass_preserving_action` at all, so none of what follows was previously
covered by any test.

Reproduction script and raw output: `blur_report_assets/quantify_mass_preserving.py`,
`blur_report_assets/mass_preserving_quantification_results.json`.

### D.2 `jacobian_determinant` is miscalibrated at zero deformation, and its fold-sign convention is inverted from the usual literature convention

Evaluating `jacobian_determinant` ([deform.py:30-68](../../operators/lddmm/deform.py#L30-L68)) on the identity field (zero
deformation) should return exactly 1 everywhere — the true Jacobian determinant of the identity map.
Instead, on this fixture's `128x128` grid it returns a near-constant **-4.063** away from the domain
edge:

```
identity field, interior (3px boundary ring excluded): min -4.0633, max -4.0632, mean -4.0632
```

This traces to the central-difference line itself
(`Dphi[..., i] = s * (phi.roll(shifts=1, dims=i+1) - phi.roll(shifts=-1, dims=i+1)) / (2 * spacing)`,
[deform.py:58](../../operators/lddmm/deform.py#L58)): with the default `extent` (unit spacing, since no `extent` kwarg is
passed anywhere in this test fixture or in `quantify_blur.py`), this reduces to a first-difference
step of `2s/(s-1)` per axis instead of the `1` a correctly normalized central difference of the
identity map would give (`s` = grid size along that axis, 128 here). The resulting per-axis diagonal
entries of `Dphi` are `≈ -2.0157` instead of `1`, so `det ≈ -(2.0157)^2 ≈ -4.063` instead of `1`. This
was confirmed to scale as expected — `uniform_scale_det_interior_mean_by_c` for `phi = c * id` measured
`-1.016` (c=0.5), `-4.063` (c=1.0), `-16.253` (c=2.0), i.e. it scales as `c^2` around the same wrong
baseline rather than being pinned to 1 at `c=1`. **This bug is entirely independent of the swirl field
or field composition (section A/B) — it is present for zero deformation.**

A second, separate property was checked because it changes how "folding" must be read out of this
function: which *sign* actually indicates folding here. A single-axis mirror flip
(`phi_x -> -phi_x`), the textbook orientation-*reversing* map, was measured against the identity
field's sign:

| Field | Interior mean determinant | Orientation |
|---|---:|---|
| Identity (`c=1`) | -4.063 | preserving (by construction) |
| Uniform scale (`c=2`, still preserving) | -16.253 | preserving |
| Mirror flip (orientation-reversing) | **+4.063** | reversing |

So in this codebase's convention, the non-folded/orientation-preserving sign is **negative**, and
folding/orientation-reversal flips it **positive** — the opposite of the customary convention in the
diffeomorphic-registration literature (e.g. Dalca et al. 2019, cited in C.5, which reports "folding
voxels" as `det ≤ 0`). This is a direct consequence of the reversed subtraction order in the
`roll(shifts=1) - roll(shifts=-1)` line above (a standard forward-minus-backward central difference
would be `roll(shifts=-1) - roll(shifts=1)`); it was not assumed, it was verified with the mirror-flip
test above. **Anyone reading `jacobian_determinant`'s sign directly (e.g. to build a folding check per
suggestion C.5) needs `det > 0`, not `det < 0`, given the current code** — all fold-fraction numbers in
D.3 below use this codebase's convention (`det > 0` = folded), not the literature's.

### D.3 Folding does occur in the swirl fixture, grows with more compositions, and is masked by `abs()`

Using the `det > 0` convention established in D.2, and excluding a 3-pixel boundary ring (see below),
the fraction of interior pixels with a folded/orientation-reversed local Jacobian in the euler/N=7
primary fixture:

| Frame | Fold fraction (interior, `det > 0`) | `phi` samples landing outside `[-1, 1]` |
|------:|---------------------------------------:|-------------------------------------------:|
| 1 | 0.28% | 10.5% |
| 2 | 2.78% | 20.1% |
| 3 | 7.83% | 29.6% |
| 4 | 12.76% | 39.1% |
| 5 | 17.99% | 48.5% |
| 6 | 22.71% | 57.6% |
| 7 (last) | **27.59%** | **66.4%** |

Folding grows monotonically, reaching over a quarter of the interior domain by the last frame — this
is consistent with B.2's independent observation (point 4) of "jagged, high-curvature edges" and
structured aliasing appearing in frames 5-7, and with the swirl field's rotational singularity at the
image center (clamped, per B.1) being exactly the kind of local curvature that field self-composition
erodes into folds. The outer boundary ring was excluded because `jacobian_determinant`'s
`phi.roll(shifts=±1, ...)` wraps circularly at the array edge
([confirmed by PyTorch's documented behavior](https://docs.pytorch.org/docs/stable/generated/torch.roll.html):
"elements that are shifted beyond the last position are re-introduced at the first position"), which
produces spurious finite-difference spikes unrelated to the deformation (observed directly: raw
determinant values up to **+255.98** and **-16127.0** on the outermost ring of the N=7 last frame,
versus a `±4-848` range in the interior). This boundary artifact is real but not the main story: of the
pixels flagging positive (folded) in the last frame, only 8.2% sit exactly on the outer ring (which is
itself only ~3% of all pixels, so the ring is over-represented — a genuine, separate artifact worth
noting), while the rest span the full frame including the swirl center (minimum distance from center
among folded pixels: 0 px; mean: 55 px) — i.e. most of the measured folding is a property of the
field, not the roll wraparound.

`mass_preserving_action` takes `torch.abs()` of this signed determinant unconditionally
([deform.py:90](../../operators/lddmm/deform.py#L90)) before scaling the image. The task brief's suggestion that this
`abs()` silences the fold signal rather than reporting it is confirmed directly here, not just in
principle: by frame 7, over a quarter of the interior domain is folded, and `mass_preserving_action`
rescales every one of those pixels by a positive multiplier indistinguishable from a well-behaved,
non-folded region — there is no signal left downstream that would let a caller detect that ~28% of the
frame came from a topologically broken part of the map.

### D.4 Mass conservation: does `mass_preserving_action` actually hold total intensity closer to the template than `geometric_action`?

This is the actual claimed purpose of mass-preserving action per the lung-CT registration literature
cited in C.6 (Yin, Hoffman & Lin et al. 2009/2011; Gorbunova et al.) — conserving integrated
intensity/"mass" under local compression/expansion. Total image intensity (pixel sum) was measured per
frame, relative to the template, for both actions on the same `phi_k`:

![Mass conservation curves](blur_report_assets/mass_conservation_curves.png)

**Primary fixture (`magnitude=50`, matching the swirl stress-test used throughout this report):**

| Frame | `geometric_action` (% of template) | `mass_preserving_action` (% of template) |
|------:|--------------------------------------:|---------------------------------------------:|
| 1 | 93.7% | 901.6% |
| 2 | 80.3% | 674.9% |
| 3 | 69.9% | 605.6% |
| 4 | 59.3% | 570.1% |
| 5 | 48.1% | 503.2% |
| 6 | 36.3% | 433.5% |
| 7 (last) | 26.2% | 376.4% |

Neither action holds intensity anywhere close to constant, and mass-preserving is worse in absolute
deviation from 100% at every single frame — the opposite of its intended effect. Two distinct causes
were checked rather than assumed:

1. **The fixture is not actually "closed boundary" at this magnitude.** The `frac_phi_sampling_outside_domain`
   column in D.3's table shows that by frame 7, **66.4%** of `phi`'s sample coordinates fall outside
   `[-1, 1]`. `geometric_action`'s `grid_sample(..., padding_mode="zeros")`
   ([deform.py:83](../../operators/lddmm/deform.py#L83)) returns exactly 0 for every one of those samples — and multiplying
   zero by any Jacobian factor, however correct, is still zero. This means a large share of the
   intensity loss visible in `geometric_action`'s column above is mass that has genuinely left the
   finite grid, which no Jacobian-determinant correction of the kind implemented here can restore. The
   swirl magnitude used throughout this report (50) was chosen in B.1 as a deliberate stress test to
   make the composition-blur mechanism visible within `N=7` steps; that same choice means this
   particular fixture is not a fair test of mass-preservation's actual design assumption. This was
   verified directly, not inferred from the task brief's caveat about it.
2. **Independent of (1), the D.2 calibration bug inflates `mass_preserving_action` by roughly a
   constant factor even where content stays inside the domain.**

To separate these two effects, a supplementary run used a much smaller swirl magnitude (5 instead of
50, same `N=7`/euler), which keeps boundary leakage far lower (≤7.15% of samples, versus ≤66.4% at
magnitude 50) and produces **zero folding** in the interior at every frame — i.e., a regime much closer
to the "closed boundary, no folding" case mass-preservation math assumes:

| Frame | `geometric_action` (% of template) | `mass_preserving_action` (% of template) | Fold fraction | `phi` outside `[-1,1]` |
|------:|--------------------------------------:|---------------------------------------------:|---------------:|--------------------------:|
| 1 | 99.6% | 406.3% | 0.00% | 1.6% |
| 2 | 99.1% | 406.3% | 0.00% | 3.1% |
| 3 | 98.6% | 406.3% | 0.00% | 3.7% |
| 4 | 98.2% | 406.3% | 0.00% | 4.5% |
| 5 | 97.7% | 406.3% | 0.00% | 5.5% |
| 6 | 97.2% | 454.3% | 0.00% | 6.3% |
| 7 (last) | 96.8% | 576.5% | 0.00% | 7.2% |

Here, `geometric_action` **alone** already holds total intensity within 3.2% of the template through
all 7 frames, with no Jacobian correction at all — undercutting the premise that a Jacobian correction
is needed to hold mass roughly constant under this particular (mild) swirl. `mass_preserving_action`
still overshoots by 4-5.8x, and since folding is confirmed at exactly 0% throughout this run, that
overshoot cannot be attributed to folding — it is the D.2 calibration bug (an `abs()`-of-roughly--4.06
multiplier applied almost everywhere) acting essentially alone.

**Finding: in neither regime tested — the report's own stress-test fixture, nor a much milder
low-leakage/zero-fold variant added specifically to give mass-preservation a fair chance — does
`mass_preserving_action` conserve total image intensity better than plain `geometric_action`. It is
worse in both, by a large margin, and the dominant cause in both is the D.2 calibration bug rather than
anything about the swirl or composition dynamics from section B.** This directly answers the open
question left in C.6: not only was no literature found claiming mass-preservation would incidentally
fix the composition blur, but on this evidence the current implementation does not yet deliver its own
stated purpose (mass conservation) either — that appears to be a pre-existing bug in
`jacobian_determinant`, not a property of mass-preservation as a technique. This report does not
attempt to fix it (out of scope, per the brief), but the specific line identified in D.2
([deform.py:58](../../operators/lddmm/deform.py#L58)) is where that fix would need to start.

### D.5 Effect on the composition-blur trend from section B

The same three B.2 sharpness metrics were computed for both actions, per frame, with a 3-pixel
boundary ring excluded (to keep the D.3 roll-wraparound artifact from dominating a metric it has
nothing to do with):

![Sharpness metric curves, geometric vs mass-preserving](blur_report_assets/mass_preserving_sharpness_curves.png)

| Frame | `geometric_action` grad. energy (% of template) | `mass_preserving_action` grad. energy (% of template) |
|------:|---------------------------------------------------:|-----------------------------------------------------------:|
| 1 | 59.8% | 1,540.9% |
| 2 | 49.0% | 2,126.0% |
| 3 | 44.2% | 5,700.3% |
| 4 | 41.9% | 9,783.5% |
| 5 | 42.4% | 18,046.4% |
| 6 | 39.0% | 26,509.7% |
| 7 (last) | 32.6% | **39,927.2%** |

`geometric_action`'s trend matches B.2 (monotonic decline, same order of magnitude — small numeric
differences from B.2's table are just the 3px interior crop used here). `mass_preserving_action` does
**not** track this trend at all: rather than declining like a blur metric or holding flat like a
"no effect" result, it grows by nearly 4 orders of magnitude across the sequence, in the *opposite*
direction from `geometric_action`. Two contributions to this were distinguished, again by direct
measurement rather than assumption:

- Even in the zero-fold, low-leakage supplementary fixture (D.4's `magnitude=5` run), gradient energy
  for `mass_preserving_action` is already inflated to ~1,000-1,300% of template at every frame (e.g.
  1,024% at frame 1, 1,099% at frame 7) — roughly consistent with squaring the D.2 calibration factor
  (a near-uniform `~4x` intensity multiplier feeding into a squared-gradient metric lands in
  this range), and does **not** grow much further across frames, since there is no folding here to add
  more.
- In the primary (`magnitude=50`) fixture, the inflation keeps *growing* through the sequence (1,541%
  to 39,927%, a further ~26x on top of the baseline inflation) in step with the growing fold fraction
  from D.3 (0.28% to 27.6%) — consistent with sign-crossing det discontinuities at fold boundaries
  creating sharp, spurious edges that a gradient/Laplacian-based metric reads as extra "detail." This is
  visible directly in the frame gallery below as a bright spurious line running through the swirl
  center that is not present in the `geometric_action` frames:

![Frame gallery, geometric vs mass-preserving](blur_report_assets/mass_preserving_frame_gallery.png)

**Finding: `mass_preserving_action` does not reduce the composition blur from section B, is not
orthogonal to it, and is not simply "the same blur, scaled" — it actively manufactures a large,
growing amount of spurious high-frequency content that has nothing to do with real image detail.**
Unlike B.2 point 4's aliasing/ringing (already flagged there as a harder failure mode than plain blur
because it can look like fabricated structure), this is more severe still: an order-of-magnitude
brightness/gradient explosion that would be immediately visually obvious rather than subtly misleading
— but it means naively swapping `action="geometric"` for `action="mass_preserving"` to "fix" B would,
on this evidence, make the output substantially worse, not better.

**What was not verified:** whether a *correctly calibrated* Jacobian determinant (i.e. with the D.2
bug fixed) would still show this same growing-with-folding inflation, a smaller version of it, or none
at all. That would require a fix to `jacobian_determinant`, which is out of scope for this
measurement-only report (the brief explicitly asks that `deform.py` not be modified) — flagged here as
an open question rather than assumed one way or the other.

---

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
- [`torch.roll` documentation](https://docs.pytorch.org/docs/stable/generated/torch.roll.html) — confirms circular wraparound at array edges, used in D.3 to explain the boundary-ring artifact in `jacobian_determinant`.
