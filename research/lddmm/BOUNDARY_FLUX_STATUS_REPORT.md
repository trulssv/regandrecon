# Status report: mass/content loss at open (net-flux) domain boundaries in `FlowDeformationOperator`

**Scope:** `operators/lddmm/deform.py` — `GroupAction.geometric_action`, `GroupAction.jacobian_determinant`,
`GroupAction.mass_preserving_action`, `VelocityIntegrator` (euler integration via repeated composition).
**Trigger:** Real respiratory-motion CT sequences do not have a closed/static domain: during exhalation the
diaphragm rises, producing a genuine net flow of tissue across the domain boundary along the `D` (depth /
superior-inferior) axis — not merely local compression/expansion inside a fixed FOV. This is a distinct
scenario from [`BLUR_STATUS_REPORT.md`](BLUR_STATUS_REPORT.md), which diagnoses blur from repeated bilinear
field self-composition and explicitly does not touch boundary/flux behavior.
**Status:** Root cause identified and quantified on controlled synthetic phantoms/fields that reproduce the
mechanism without requiring real CT data. Not fixed — diagnosis + options only, matching the scope of the
existing blur report. `deform.py` was not modified.

---

## A. What is happening

Three separate, verified mechanisms combine to make `mass_preserving_action` — and, less obviously,
`geometric_action` — behave incorrectly once real anatomy sits at or crosses a domain boundary that is
genuinely open (net flux), rather than closed (motion that vanishes at the edge).

### A.1 `geometric_action`'s `padding_mode="zeros"`: correct for exiting content, wrong for entering content

`geometric_action` ([deform.py:83](../../operators/lddmm/deform.py#L83)) computes `deformed = grid_sample(x, phi, ...,
padding_mode="zeros")`. Per the class docstring, this implements $x\circ\phi^{-1}$: for every *output*
location $p$, it samples the *template* $x$ at source location $\phi(p)$. When $\phi(p)$ lies outside
$[-1,1]$ — i.e. the model needs content from outside the template's original support — PyTorch's
`padding_mode="zeros"` returns exactly 0, decaying smoothly to 0 only within the last source voxel before
the edge and hard-zero beyond it. We confirmed this directly (`quantify_mass_preserving_flux.py`,
`confirm_zero_padding_semantics`): sampling a constant-500 region at increasing out-of-range coordinates
gives `500, 500, 500, 400.0, 200.0, 0.0002, 0.0` for `padding_mode="zeros"` vs. a flat `500, 500, 500, 500,
500, 500, 500` for `padding_mode="border"`.

This has two directionally opposite consequences that the current code does not distinguish:

- **Content moving out of the FOV** (real tissue displaced past the boundary): rendering it as 0 is
  *physically correct* — that tissue genuinely left the imaged volume.
- **Content that should move into the FOV from previously-unimaged tissue** (e.g. abdominal viscera coming
  into view as the diaphragm rises): there is no way for `padding_mode="zeros"` — or any padding mode
  applied to a template that never contained that tissue — to be correct here. Zero is a specific, wrong
  value (it renders incoming tissue as air/vacuum) precisely because the true density is unknown, not
  because it is actually zero. This is a fundamental data-availability limitation, not a bug that can be
  patched by choosing a different `padding_mode` on `grid_sample` alone (see C.1/C.2 for what toolkits do
  instead).

Which of these two cases dominates depends on the direction of the flow and the phantom's position; B.3
walks through a concrete, measured example.

### A.2 `jacobian_determinant`'s `torch.roll`-based boundary stencil is broken — but *not* specifically because of net flux

`jacobian_determinant` ([deform.py:30-68](../../operators/lddmm/deform.py#L30-L68)) estimates each spatial derivative with
`phi.roll(shifts=1, dims=i+1) - phi.roll(shifts=-1, dims=i+1)` ([deform.py:58](../../operators/lddmm/deform.py#L58)).
`torch.roll` is circular: at row `i=0`, the "`i-1`" neighbor it uses is `phi` at row `H-1` — the *opposite*
edge of the array — not an extrapolated exterior value. This is only a valid finite-difference stencil if
`phi` is periodic across that axis. It is not: `phi` is built from a `linspace(-1, 1)` identity grid, which
has a hard, ~2-unit (normalized) discontinuity from one edge to the other.

We isolated this precisely (`quantify_boundary_jacobian.py`): a reference implementation,
`jacobian_determinant_replicate`, is byte-for-byte identical to the original formula except the boundary
neighbor is obtained by clamping the index (edge-replicated / Neumann-style) instead of wrapping. Interior
values (rows/cols away from every edge) match exactly between the two by construction — any discrepancy
elsewhere is purely the wraparound artifact, not a property of the field.

**Result: the corruption is large, and it is present for every field we tested, including the pure identity
map (zero velocity).** Mean |roll − replicate| error at the boundary row, restricted to interior columns to
avoid conflating the two axes' independent wraparounds:

| Field | Mean \|error\| at boundary row | Max \|error\| at boundary row | Interior error (both axes) |
|---|---:|---:|---:|
| Identity (zero velocity) | 130.0 | 130.0 | 0.0 (exact) |
| Piston (net-flux ramp) | 108.9 | 108.9 | 0.0 (exact) |
| Swirl (existing `test_deform.py` fixture) | 98.3 | 125.8 | 0.0 (exact) |

This directly answers a question this investigation set out to check: **the existing swirl fixture used by
`BLUR_STATUS_REPORT.md` does *not* accidentally avoid this bug.** Its boundary error (mean 98.3, max 125.8)
is the same order of magnitude as both the zero-velocity identity map and the net-flux piston field. The
dominant source of the error is rolling the *identity grid itself* (which is non-periodic by construction),
not the specific velocity field riding on top of it — so this is a **general defect in
`jacobian_determinant`, not one that is unique to open/net-flux boundaries.** (See C.4 for a distinct,
separately-verified scale/sign defect found while sanity-checking this: even the fully-corrected, non-wrapped
interior Jacobian determinant of the identity map does not evaluate to the mathematically required `1.0` —
it evaluates to `-0.33` at the phantom test's real extent/shape (450 mm / 128 px), and to values around `-4`
to `-5` at unit-extent/smaller shapes tested. That is a second, independent defect, orthogonal to the
roll/boundary issue, and out of scope for this report beyond flagging it here since it inflates
`mass_preserving_action`'s output by a roughly constant multiplicative factor regardless of boundary
condition — see B.1.)

**Why this still matters more for the open-boundary case in practice**, even though the bug itself is
boundary-condition-agnostic: the *consequence* of a corrupted per-pixel Jacobian multiplier is proportional
to how much real image intensity sits at the corrupted location. A closed/vanishing-at-the-edge field can
still have real anatomy away from the edge, where the multiplier is fine. A genuinely open boundary (e.g. the
diaphragm/lung-base interface) puts clinically important tissue *exactly* where this defect is largest by
construction — there is no "safe distance from the edge" available to the physiology.

### A.3 `mass_preserving_action`'s conservation identity presumes a closed domain — and breaks structurally under real flux, independent of A.2

`mass_preserving_action` = `geometric_action(x, phi) * |det Dφ⁻¹|` ([deform.py:87-93](../../operators/lddmm/deform.py#L87-L93)) is
the discretized change-of-variables identity $\int_\Omega x(\phi^{-1}(p))\,|D\phi^{-1}(p)|\,dp =
\int_\Omega x(q)\,dq$, which holds when $\phi:\Omega\to\Omega$ is a diffeomorphism of the domain **onto
itself** — i.e. a closed system, exactly what CT-lung mass-preserving registration literature (Yin et al.
2009; Gorbunova et al.) formulates and validates against. The moment the *true* motion is not confined to
$\Omega$ (some of $\phi^{-1}(p)$ for $p\in\Omega$ genuinely lies outside the imaged volume, or vice versa),
the premise of the identity is violated *before any implementation bug is even considered*: no per-pixel
multiplicative correction, correctly computed or not, can restore intensity that was never sampled from
anywhere. This is the mathematical reason B.2/B.3 show mass-preserving action's correction only *partially*
compensates non-conservation for the open field and degrades further as flux increases, whereas it is
essentially exact for the closed field at the same peak velocity magnitude.

### A.4 A boundary-specific extension of the existing composition-blur mechanism: reachable-source-range collapse

The project's real integration path (`euler`, `DEFORM_PARAMS["integration"]="euler"`,
[deform.py:293-320](../../operators/lddmm/deform.py#L293-L320)) builds `phi_k` by composing a small per-step field `phi0 = id -
v/N` with itself `k` times via `_compose` ([deform.py:187-210](../../operators/lddmm/deform.py#L187-L210)) — the same mechanism
`BLUR_STATUS_REPORT.md` identifies as progressively eroding field detail. We found a boundary-specific
consequence of this that the blur report does not cover: **for a field with a real, monotonically-increasing
component reaching a boundary (net flux), the *maximum source coordinate reachable by any output pixel*
contracts away from that boundary with each additional composition step**, at a rate set by the field's
magnitude there. Empirically (`quantify_mass_preserving_flux.py`, `reach_profile`), for a 100 mm peak
excursion the reachable source-y drops from `0.96` (frame 1) to `0.59` (frame 7) — monotonically, every
frame — while the equivalent closed field's reachable range stays pinned at exactly `1.0` throughout (its
value at the boundary is zero, so composition never needs to "reach past" the edge in the first place). Once
the reachable range's edge passes the physical location of real anatomy sitting near the boundary (in our
phantom, source-y ≈ `0.70`), that anatomy's contribution to total intensity **collapses to exactly zero**,
not a gradual fade (B.3). We verified this is not a `padding_mode` artifact of `_compose` specifically:
re-running the same composition with `padding_mode="border"`, `"zeros"`, and `"reflection"` gives an
identical reachable-range trajectory to 3 decimal places, because no individual `_compose` call ever
actually samples out-of-bounds in this regime — the contraction is a property of *repeatedly composing a
boundary-approaching field*, not of how any single composition step handles its own edges. This makes it a
new, boundary-specific manifestation of the general mechanism `BLUR_STATUS_REPORT.md` already flagged
(repeated bilinear self-composition erodes information) — except here the erosion is not "blur," it is
**total, silent, unrecoverable loss of a whole anatomical region** from the output sequence.

---

## B. Quantified impact

### B.1 Method

Real CT volumes were not available in this environment (same constraint as the blur report). Two
purpose-built synthetic tests isolate the boundary/flux mechanism from CT-specific confounds, and both run
through the *unmodified* production code path (`FlowDeformationOperator.forward(..., superres=True)`) at
`N=7`, `integration="euler"` — the project's real `DEFORM_PARAMS` — and, for the second test, the project's
real 2D physical extent `((0,450),(0,450))` mm.

- **`quantify_boundary_jacobian.py`** isolates the `torch.roll` artifact in `jacobian_determinant` itself
  (A.2), independent of `mass_preserving_action`, on an identity map, a net-flux "piston" field, and the
  existing swirl fixture.
- **`quantify_mass_preserving_flux.py`** runs a bar phantom (rows 108–127 of a 128×128 image, near the
  bottom edge, intensity 1000 HU-like units, zero elsewhere) through the real
  `FlowDeformationOperator`, comparing two velocity fields matched in **peak magnitude** but differing in
  **where that peak sits relative to the boundary**:
  - `closed`: $v_y(i) = V_0 \sin(\pi i/(H-1))$ — zero at *both* edges (no flux crossing either boundary;
    peak compression/expansion happens mid-domain). This is the scenario `mass_preserving_action` is
    designed for and the only scenario the existing swirl fixture exercises.
  - `open`: $v_y(i) = V_0\, i/(H-1)$ — zero at the top edge, ramping to the full magnitude $V_0$ *at* the
    bottom edge. This models a diaphragm-like open boundary: real, nonzero velocity penetrating the domain
    edge, increasing toward it — matching the diaphragm-rise scenario in the trigger.

  $V_0$ was swept across literature-reported diaphragm excursion: **20 mm** (quiet tidal breathing,
  reported range ≈15–23 mm) and **60 mm** (deep/forced breathing, reported range ≈53–69 mm; see References),
  plus a **100 mm** stress-test point beyond the literature range, in the same spirit as the blur report's
  large-swirl stress test.

### B.2 Total intensity per frame: closed vs. open, geometric vs. mass-preserving

![Total intensity vs frame](boundary_flux_report_assets/total_intensity_vs_frame.png)

| Excursion | Field | Action | Frame 1 → Frame 7 (% of template) | Frame1→7 relative drop |
|---|---|---|---|---:|
| 20 mm | closed | geometric | 98.2% → 87.7% | 10.7% |
| 20 mm | closed | mass_preserving | 265.3% → 264.9% | **0.15%** |
| 20 mm | open | geometric | 96.5% → 75.4% | 21.9% |
| 20 mm | open | mass_preserving | 261.4% → 238.5% | **8.76%** |
| 60 mm | closed | geometric | 94.7% → 68.0% | 28.2% |
| 60 mm | closed | mass_preserving | 265.2% → 263.9% | **0.49%** |
| 60 mm | open | geometric | 89.4% → 22.0% | 75.4% |
| 60 mm | open | mass_preserving | 253.6% → 187.6% | **26.0%** |
| 100 mm (stress) | closed | mass_preserving | 265.0% → 262.8% | 0.83% |
| 100 mm (stress) | open | mass_preserving | 245.7% → **0.0%** | 100% |

Raw numbers: `boundary_flux_report_assets/mass_preserving_flux_results.json`.

Key findings:

1. **The absolute percentages are not near 100% even for the closed field, for a reason unrelated to
   boundary flux** (A.2's aside): a general scale/sign defect inflates `mass_preserving_action`'s output by
   a roughly constant multiplicative factor (~2.6×) regardless of scenario. This should be fixed on its own
   merits but is not what this report is about — the useful signal is the *trend*, not the offset.
2. **The trend cleanly separates open from closed.** At the same peak velocity, `mass_preserving_action`'s
   output stays essentially flat frame-over-frame for the closed field (0.15–0.83% drift across the entire
   20–100 mm range) but decays substantially and increasingly for the open field: 8.8% (20 mm, quiet
   breathing) → 26.0% (60 mm, deep breathing) → **100% (complete loss, 100 mm stress test)**. Mass
   preservation is not merely "less accurate" under real flux — it degrades monotonically with the flux
   magnitude and can fail completely.
3. **`mass_preserving_action` does substantially reduce raw geometric loss for the open field** (e.g. at 60
   mm, raw geometric loses 75.4% vs. mass-preserving's 26.0%), confirming the Jacobian correction is doing
   real, useful work — it is just structurally incomplete under genuine flux (A.3), not simply broken.
4. **At 100 mm, both `geometric` and `mass_preserving` collapse to exactly 0.0%** — not a floor/plateau, an
   exact zero. This is the reach-contraction mechanism in A.4/B.3, not a gradual interpolation loss.

### B.3 Mechanism of the collapse: reachable source range vs. phantom location

![Reach contraction](boundary_flux_report_assets/reach_contraction.png)

The maximum source-y coordinate reachable by *any* output pixel contracts every frame for the open field
(0.96 → 0.75 over 7 frames at 60 mm; 0.96 → 0.59 at 100 mm) while staying pinned at exactly 1.0 for the
closed field throughout. The phantom's near edge sits at source-y ≈ 0.70. At 100 mm, the reachable range
drops below that threshold between frames 5 and 6 — and total intensity drops from 5.2% to exactly 0.0% at
that same transition (B.2 raw JSON), confirming the mechanism, not coincidence.

![Frame gallery, open field, 60mm](boundary_flux_report_assets/frame_gallery_open_60mm.png)

At 60 mm the phantom visibly shrinks and dims frame-over-frame in the `mass_preserving` sequence — the bar
narrows toward the bottom edge and does not visually "recover" the way the closed case does, consistent with
the intensity numbers in B.2. **No individual `grid_sample` call in this experiment was found to sample a
literally out-of-range coordinate** (`oob_fraction_per_frame` is `0.0` at every frame, every excursion
tested up to 100 mm) — the loss is fully accounted for by the reach-contraction mechanism (A.4), not by raw
`padding_mode="zeros"` clipping in this specific regime. A.1's zero-padding mechanism is real and separately
confirmed (via the minimal direct `grid_sample` test in B.1), but at the magnitudes tested here it did not
turn out to be the dominant contributor to the measured loss — the composition-driven reach contraction was.
This distinction matters for anyone deciding where to spend fixing effort (see C).

---

## C. Ways to combat this

As with the blur report, all options below are evaluated against the requirement that **information should
not be lost** for a clinically applicable pipeline. This is an active area of the registration literature;
citations are given per point, and points where no direct supporting source was found are labeled as such.

1. **Treat the open boundary explicitly with an adaptive/local boundary condition, rather than the current
   implicit global one.** The current code applies one fixed rule (`padding_mode="zeros"` everywhere) with
   no notion of "this edge of the domain is physically open." Inacio et al., *"Adaptive local boundary
   conditions to improve Deformable Image Registration,"* propose a locally adaptive Robin-type condition
   that balances Dirichlet and Neumann behavior **depending on the incoming/outgoing flow at each boundary
   point** — exactly the distinction A.1 draws between exiting and entering content — and report up to 12%
   (mean 4%) target-registration-error improvement over fixed homogeneous conditions on CT thorax and
   abdominal registration. This is the most directly applicable citation found for the core problem
   diagnosed here.
2. **For the specific case of lung/diaphragm sliding motion, use a registration formulation designed for
   discontinuous boundary motion rather than a single global diffeomorphism.** Risser, Vialard et al.,
   *"Piecewise-diffeomorphic image registration: Application to the motion estimation between 3D CT lung
   images with sliding conditions,"* Medical Image Analysis 17(2), 2013, formalize LDDMM with explicit
   sliding conditions at organ boundaries, validated on the EMPIRE10 lung CT challenge; independent
   physiological measurements confirm lung motion is smooth internally but genuinely discontinuous
   (sliding) at the pleural/diaphragmatic boundary (PMC10538991, "Role of lung lobar sliding on parenchymal
   distortion during breathing"), which is a different failure mode from — but adjacent to — the pure net
   flux modeled in this report's synthetic test, and worth distinguishing when choosing a fix.
3. **Fix `jacobian_determinant`'s boundary stencil independent of any flux-specific fix (A.2), since it is
   a general defect, not just a flux-specific one.** Replace the circular `torch.roll` with a one-sided
   difference or edge-replicated ("Neumann"/`padding_mode="border"`-equivalent) stencil at the domain edge —
   we already built and validated exactly this as `jacobian_determinant_replicate` in
   `boundary_flux_report_assets/quantify_boundary_jacobian.py` (kept out of `deform.py` per this report's
   diagnosis-only scope). This alone would not fix A.3's structural mass-loss under genuine flux (A.3 is a
   mathematical property of the change-of-variables identity, not an implementation bug), but it would
   remove a large (order-of-magnitude), verified, boundary-agnostic corruption that currently affects *every*
   use of `mass_preserving_action`, including the existing closed-boundary swirl fixture.
4. **Adopt established toolkits' more conservative default for content sourced from outside the domain.**
   MONAI's `Warp` block ([`monai/networks/blocks/warp.py`](https://github.com/Project-MONAI/MONAI/blob/main/monai/networks/blocks/warp.py))
   defaults `padding_mode` to **`"border"`**, not `"zeros"` — i.e. content sampled from just outside the FOV
   repeats the nearest known edge value rather than injecting a fabricated zero/air value. This is still not
   a correct answer for genuinely unimaged tissue (A.1), but it is a materially less wrong one than the
   current hardcoded `padding_mode="zeros"` at [deform.py:83](../../operators/lddmm/deform.py#L83), and is a one-line, low-risk
   change to consider. ITK's `ResampleImageFilter` defaults to a fixed `DefaultPixelValue` of 0 (same
   convention as this codebase) but explicitly supports swapping in a custom `Extrapolator` — confirming
   zero-padding-by-default is a common, acknowledged-as-imperfect convention across toolkits, not a defect
   unique to this codebase, but one every toolkit treats as configurable rather than fixed.
5. **Extend the field of view / accept that true mass conservation across an open boundary requires data
   this pipeline does not have**, if clinically accurate absolute mass accounting at the boundary itself is
   required. No mechanism internal to `deform.py` — Jacobian-corrected or not — can recover tissue density
   for anatomy that was never imaged (A.3). Mang & Ruthotto's *"A Lagrangian Gauss-Newton-Krylov Solver for
   Mass- and Intensity-Preserving Diffeomorphic Image Registration,"* SIAM J. Sci. Comput. 39(5), 2017,
   formulate both the transport (intensity-preserving) and continuity (mass-preserving) equations solved via
   a Lagrangian hyperbolic PDE solver; their formulation, like the classic lung-CT mass-preserving papers
   (Yin et al. 2009; Gorbunova et al.), assumes registration within a single fixed, closed domain and does
   not address flux through the domain boundary — no source was found (in this literature or otherwise)
   claiming a Jacobian-based mass-preserving correction can be made exact under genuine open-boundary flux;
   this is flagged as an apparent structural limitation of the whole formulation family, not something this
   report found a citation resolving, rather than a settled claim.
6. **Separately from the boundary-flux issue, decouple the number of Euler/scaling-and-squaring steps `N`
   from the total flow magnitude when a field has a boundary-approaching component.** B.3's reach-contraction
   mechanism gets strictly worse with more composition steps for a fixed total displacement (same structural
   reason `BLUR_STATUS_REPORT.md` C.1 recommends re-integrating from scratch per output frame rather than
   composing forward) — an additional, boundary-specific reason to prefer that report's option 1
   (re-integrate `t·v` at each desired `t` rather than repeatedly composing) when the estimated velocity
   field has any component reaching a domain edge, on top of the detail-preservation reason already given
   there.

### References

- Inacio, Lafitte, Facq, Poignard, Denis de Senneville, ["Adaptive local boundary conditions to improve Deformable Image Registration,"](https://arxiv.org/abs/2405.12791) arXiv:2405.12791, 2024.
- Risser, Vialard, Wolz, Murgasova, Holm, Rueckert, ["Piecewise-diffeomorphic image registration: Application to the motion estimation between 3D CT lung images with sliding conditions,"](https://www.sciencedirect.com/science/article/abs/pii/S1361841512001466) Medical Image Analysis, 17(2), 2013.
- ["Role of lung lobar sliding on parenchymal distortion during breathing,"](https://pmc.ncbi.nlm.nih.gov/articles/PMC10538991/) PMC10538991.
- Mang, Ruthotto, ["A Lagrangian Gauss-Newton-Krylov Solver for Mass- and Intensity-Preserving Diffeomorphic Image Registration,"](https://arxiv.org/abs/1703.04446) SIAM J. Sci. Comput., 39(5), B860-B885, 2017.
- ["Mass preserving nonrigid registration of CT lung images using cubic B-spline,"](https://pubmed.ncbi.nlm.nih.gov/19810495/) Yin, Hoffman, Lin, Medical Physics, 2009.
- ["Mass preserving image registration for lung CT,"](https://www.sciencedirect.com/science/article/abs/pii/S1361841511001617) Gorbunova et al., Medical Image Analysis, 2011.
- MONAI, [`monai/networks/blocks/warp.py`](https://github.com/Project-MONAI/MONAI/blob/main/monai/networks/blocks/warp.py) — `Warp` block, `padding_mode` default `"border"`.
- ITK, [`itk::ResampleImageFilter` documentation](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1ResampleImageFilter.html) — default pixel value 0 outside the domain, configurable `Extrapolator`.
- PyTorch, [`torch.nn.functional.grid_sample` documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) — `padding_mode` semantics (`"zeros"`, `"border"`, `"reflection"`).
- Diaphragmatic excursion reference ranges (quiet breathing ≈1.5–2.3 cm; deep breathing ≈5.3–6.9 cm, ultrasound M-mode and fluoroscopy studies) — used to set the 20 mm / 60 mm test magnitudes in B.1.
