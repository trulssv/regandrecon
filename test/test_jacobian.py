"""
Standardized correctness checks for GroupAction.jacobian_determinant, in 2D and 3D.

Each case builds a deformation field phi whose Jacobian determinant is known in closed
form (identity, uniform scale, anisotropic scale, an orientation-reversing mirror flip,
and a pure translation), via VelocityIntegrator's own identity-grid construction -- so
the test tracks whichever channel convention deform.py currently uses rather than
hardcoding an assumption about it -- and checks jacobian_determinant(phi) against the
closed-form answer.

3D cases deliberately use an anisotropic shape (D != H != W) and anisotropic extent
(a different physical size per axis), and include a mirror-flip case. This is not
incidental: a bug that pairs each axis's derivative with the wrong physical extent, or
that only flips the determinant's sign in odd dimensions, will pass a naive isotropic
2D-only identity check and still be wrong -- both of those have actually happened during
development of this function. See BLUR_STATUS_REPORT.md / BOUNDARY_FLUX_STATUS_REPORT.md
in research/lddmm/ for the history.

Run directly: python test/test_jacobian.py
"""
import sys

import torch

from operators.lddmm.deform import GroupAction, VelocityIntegrator

# Interior points (away from every domain edge) should match the closed-form answer
# tightly -- central differences on these synthetic (piecewise-linear) fields are exact
# up to floating point error.
INTERIOR_ATOL = 1e-3

# The boundary treatment (whatever it currently is -- replicate-padding, one-sided
# differencing, circular wraparound, ...) is checked with a looser, explicit tolerance
# rather than being skipped. A reasonable approximation (e.g. a one-sided derivative
# that comes out a constant factor off from the true value) stays within this band; a
# structurally broken one (e.g. circular torch.roll wraparound pulling in the *opposite*
# edge of the array as a neighbor) does not, and should be caught here rather than only
# in a downstream report.
BOUNDARY_ATOL = 0.6


def _interior(det: torch.Tensor, ndim: int) -> torch.Tensor:
    if ndim == 2:
        return det[:, 1:-1, 1:-1]
    return det[:, 1:-1, 1:-1, 1:-1]


def _boundary_slice(det: torch.Tensor, ndim: int) -> torch.Tensor:
    """One full boundary face (axis-0 = 0), excluding corners/edges shared with other faces."""
    if ndim == 2:
        return det[:, 0, 1:-1]
    return det[:, 0, 1:-1, 1:-1]


def run_case(name: str, shape: tuple, extent: tuple, transform, expected_det: float) -> dict:
    ndim = len(shape)
    vi = VelocityIntegrator(N=1, integration="euler", shape=shape)
    assert vi.id is not None
    id_grid = vi.id.unsqueeze(0)  # (1, *shape, ndim), whatever channel convention deform.py uses today
    phi = transform(id_grid)

    ga = GroupAction(action="geometric")

    try:
        det = ga.jacobian_determinant(phi)
    except Exception as exc:  # noqa: BLE001 -- we want to report *any* crash as a failing case, not stop the suite
        return {"name": name, "status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}

    interior = _interior(det, ndim)
    boundary = _boundary_slice(det, ndim)

    expected_t = torch.full_like(interior, expected_det)
    interior_ok = torch.allclose(interior, expected_t, atol=INTERIOR_ATOL, rtol=1e-3)

    scale = max(abs(expected_det), 1.0)
    boundary_ok = bool(((boundary - expected_det).abs() <= BOUNDARY_ATOL * scale).all())

    return {
        "name": name,
        "status": "PASS" if (interior_ok and boundary_ok) else "FAIL",
        "expected": expected_det,
        "interior_min": interior.min().item(),
        "interior_max": interior.max().item(),
        "interior_ok": interior_ok,
        "boundary_sample": [round(v, 4) for v in boundary.flatten()[:4].tolist()],
        "boundary_ok": boundary_ok,
    }


def build_cases() -> list:
    cases = []

    # ---------------- 2D ----------------
    shape_2d = (48, 64)                            # anisotropic shape (H != W)
    extent_2d = ((0.0, 300.0), (0.0, 900.0))        # anisotropic physical extent

    cases.append(("2D identity", shape_2d, extent_2d, lambda g: g.clone(), 1.0))
    cases.append(("2D uniform scale x1.5", shape_2d, extent_2d, lambda g: g * 1.5, 1.5 ** 2))
    cases.append((
        "2D anisotropic scale (ch0 x2, ch1 x3)", shape_2d, extent_2d,
        lambda g: g * torch.tensor([2.0, 3.0]), 2.0 * 3.0,
    ))
    cases.append((
        "2D mirror flip (orientation-reversing)", shape_2d, extent_2d,
        lambda g: g * torch.tensor([-1.0, 1.0]), -1.0,
    ))
    cases.append(("2D translation", shape_2d, extent_2d, lambda g: g + 0.2, 1.0))

    # ---------------- 3D ----------------
    shape_3d = (16, 20, 24)                                      # anisotropic shape (D != H != W)
    extent_3d = ((0.0, 100.0), (0.0, 300.0), (0.0, 900.0))         # anisotropic physical extent

    cases.append(("3D identity", shape_3d, extent_3d, lambda g: g.clone(), 1.0))
    cases.append(("3D uniform scale x1.5", shape_3d, extent_3d, lambda g: g * 1.5, 1.5 ** 3))
    cases.append((
        "3D anisotropic scale (ch0 x2, ch1 x3, ch2 x4)", shape_3d, extent_3d,
        lambda g: g * torch.tensor([2.0, 3.0, 4.0]), 2.0 * 3.0 * 4.0,
    ))
    cases.append((
        # Odd spatial dimensionality: a sign bug that squares away and hides itself in
        # 2D (an even power) does not necessarily hide itself in 3D (an odd power).
        "3D mirror flip (orientation-reversing)", shape_3d, extent_3d,
        lambda g: g * torch.tensor([-1.0, 1.0, 1.0]), -1.0,
    ))
    cases.append(("3D translation", shape_3d, extent_3d, lambda g: g + 0.2, 1.0))

    return cases


def main() -> int:
    results = [run_case(*case) for case in build_cases()]

    name_w = max(len(r["name"]) for r in results)
    print(f"{'case':{name_w}}  {'status':6}  {'expected':>9}  {'interior [min,max]':>22}  boundary sample (first 4)")
    for r in results:
        if r["status"] == "ERROR":
            print(f"{r['name']:{name_w}}  {'ERROR':6}  {r['detail']}")
            continue
        interior_range = f"[{r['interior_min']:.4f}, {r['interior_max']:.4f}]"
        flags = "" if r["status"] == "PASS" else f"  <-- interior_ok={r['interior_ok']} boundary_ok={r['boundary_ok']}"
        print(f"{r['name']:{name_w}}  {r['status']:6}  {r['expected']:9.4f}  {interior_range:>22}  {r['boundary_sample']}{flags}")

    n_fail = sum(1 for r in results if r["status"] != "PASS")
    print(f"\n{len(results) - n_fail}/{len(results)} cases passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
