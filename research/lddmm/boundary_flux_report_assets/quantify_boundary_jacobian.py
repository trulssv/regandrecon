"""
Quantify the `torch.roll`-based boundary artifact in `GroupAction.jacobian_determinant`
(operators/lddmm/deform.py:30-68), and check whether it is specific to open/net-flux
velocity fields or a general defect that also affects the closed-boundary swirl fixture
already used by BLUR_STATUS_REPORT.md / test/test_deform.py.

Method
------
`jacobian_determinant` estimates d(phi)/dx via a central difference built from
`phi.roll(shifts=+-1, dims=...)`. `torch.roll` is *circular*: at the domain edge, the
"neighbor" it differences against is the value from the *opposite* edge of the array,
not an extrapolated exterior value. This is only a correct finite-difference stencil if
`phi` is genuinely periodic across that axis -- it is not (phi is built from a
`linspace(-1, 1)` identity grid, which has a hard, non-periodic edge).

We isolate the effect of this specifically by building a second implementation,
`jacobian_determinant_replicate`, that is *byte-for-byte identical* to the original
formula (same scale factor, same sign convention, same everything) except that the
boundary neighbor is obtained by clamping the index (a "replicate"/Neumann-style
extension) instead of wrapping it circularly. Interior values are mathematically
identical between the two (central differences away from the boundary never touch the
wrapped/clamped index), so any difference between them isolates exactly the
roll-wraparound artifact.

We evaluate this on three fields:
  1. The pure identity map (zero velocity) -- a sanity baseline. Any discrepancy here is
     entirely an artifact, not a property of the velocity field.
  2. "piston": a synthetic field with a real net component at one boundary (ramps from
     0 to V0 across the domain), modeling a diaphragm-like open boundary.
  3. "swirl": the existing rotational velocity field from test/test_deform.py /
     BLUR_STATUS_REPORT.md, to check whether that fixture happens to avoid this bug.

Run from the repo root:
    python operators/lddmm/boundary_flux_report_assets/quantify_boundary_jacobian.py
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from operators.lddmm.deform import GroupAction, VelocityIntegrator

OUT_DIR = os.path.dirname(__file__)
torch.manual_seed(0)

H = W = 64


def jacobian_determinant_replicate(phi: torch.Tensor, extent=None) -> torch.Tensor:
    """Identical to GroupAction.jacobian_determinant, except the boundary neighbor used
    by the central-difference stencil is edge-replicated (clamped index) instead of
    obtained via circular `torch.roll`. Isolates the wraparound artifact only."""
    B = phi.shape[0]
    spatial_dims = phi.shape[1:-1]
    C = phi.shape[-1]
    extent = extent if extent else tuple((0.0, s) for s in spatial_dims)
    Dphi = torch.zeros((B, *spatial_dims, C, C))

    def edge_shift(t, shift, dim):
        n = t.shape[dim]
        idx = (torch.arange(n) + shift).clamp(0, n - 1)
        return t.index_select(dim, idx)

    for i, (s, e) in enumerate(zip(spatial_dims, extent)):
        spacing = (e[1] - e[0]) / s
        # matches phi.roll(shifts=1, dims=i+1) / phi.roll(shifts=-1, dims=i+1) semantics,
        # but with clamped (replicated) indices at the boundary instead of wraparound.
        plus = edge_shift(phi, -1, i + 1)
        minus = edge_shift(phi, 1, i + 1)
        Dphi[..., i] = s * (plus - minus) / (2 * spacing)

    return torch.det(Dphi)


def make_identity(H, W):
    y = torch.linspace(-1, 1, steps=H)
    x = torch.linspace(-1, 1, steps=W)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


def make_piston(H, W, V0=0.35):
    """Net-flux field: zero at the top edge (i=0), ramps to V0 at the bottom edge
    (i=H-1). Models a diaphragm-like open boundary: real, nonzero velocity right at
    the domain edge, not merely a large deformation somewhere in the interior."""
    i_idx = torch.linspace(0, 1, H)
    v = torch.zeros(1, H, W, 2)
    v[0, :, :, 1] = V0 * i_idx.view(H, 1)
    return v


def make_swirl(H, W, mag=0.35):
    """Same mechanism as test/test_deform.py's fixture (rotational field), rescaled to
    the same normalized-velocity magnitude as the piston field for a fair comparison."""
    coords = torch.arange(H, dtype=torch.float32)
    ii, jj = torch.meshgrid(coords - H / 2, coords - H / 2, indexing="ij")
    distance = torch.sqrt(ii**2 + jj**2).clamp(min=1)
    v = torch.zeros(1, H, W, 2)
    v[0, :, :, 0] = -mag * jj / distance
    v[0, :, :, 1] = mag * ii / distance
    return v


def band_stats(diff: torch.Tensor, band_rows: slice, band_cols: slice = slice(None)):
    band = diff[0, band_rows, band_cols]
    return {"mean_abs": float(band.abs().mean()), "max_abs": float(band.abs().max())}


def main():
    ga = GroupAction(action="geometric")
    vi = VelocityIntegrator(N=7, integration="euler", shape=(H, W))

    results = {}

    fields = {
        "identity": (make_identity(H, W), True),
        "piston_net_flux": (make_piston(H, W), False),
        "swirl_existing_fixture": (make_swirl(H, W), False),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (name, (v, is_identity)) in zip(axes, fields.items()):
        phi = v if is_identity else vi.euler(v, N=7)
        det_roll = ga.jacobian_determinant(phi)
        det_repl = jacobian_determinant_replicate(phi)
        diff = det_roll - det_repl

        # Restrict to interior columns (2:W-2) when isolating the row-boundary (H-axis)
        # effect specifically, since columns 0/W-1 are *also* corrupted independently
        # by the same roll-wraparound bug acting along the W axis -- mixing them in
        # would conflate the two edges. `interior` restricts both axes: a clean,
        # fully-unaffected baseline where roll and replicate must (and do) agree exactly.
        row0 = band_stats(diff, slice(0, 1), slice(2, W - 2))
        row_last = band_stats(diff, slice(H - 1, H), slice(2, W - 2))
        interior = band_stats(diff, slice(2, H - 2), slice(2, W - 2))

        # row-mean profile across all rows, middle column, for the figure
        mid = W // 2
        roll_profile = det_roll[0, :, mid].detach().numpy()
        repl_profile = det_repl[0, :, mid].detach().numpy()

        ax.plot(roll_profile, label="roll (as implemented)", color="crimson")
        ax.plot(repl_profile, label="edge-replicated (reference)", color="steelblue")
        ax.set_title(name)
        ax.set_xlabel("row index i (H axis)")
        ax.set_ylabel("det(D$\\phi^{-1}$)  (column W//2)")
        ax.legend(fontsize=8)
        ax.axvspan(0, 1, color="gray", alpha=0.15)
        ax.axvspan(H - 2, H - 1, color="gray", alpha=0.15)

        results[name] = {
            "row0_boundary_error": row0,
            "row_last_boundary_error": row_last,
            "interior_error(rows_2_to_H-2)": interior,
            "det_roll_row0_mean": float(det_roll[0, 0].mean()),
            "det_replicate_row0_mean": float(det_repl[0, 0].mean()),
            "det_roll_interior_mean": float(det_roll[0, H // 2].mean()),
            "det_replicate_interior_mean": float(det_repl[0, H // 2].mean()),
        }

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "jacobian_boundary_profile.png")
    plt.savefig(fig_path, dpi=130)
    plt.close(fig)

    out_json = os.path.join(OUT_DIR, "boundary_jacobian_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved figure to {fig_path}")
    print(f"Saved raw results to {out_json}")


if __name__ == "__main__":
    main()
