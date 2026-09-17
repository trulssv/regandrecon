"""
Quantify how `GroupAction.mass_preserving_action` behaves when real anatomy sits near
a moving/open domain boundary (diaphragm-like net flux), vs. the closed-domain
compression/expansion scenario it is designed for.

Reproduces the project's real inference configuration: `FlowDeformationOperator` with
N=7, integration="euler" (`DEFORM_PARAMS` in test/test_diffeomorphic_registration.py),
`extent=((0,450),(0,450))` (the project's 2D physical extent), run on a bar phantom
placed near one domain edge.

Two velocity fields, matched in peak magnitude, differing only in *where* that peak
sits relative to the boundary:
  - "closed": v = V0 * sin(pi * i / (H-1))  -- zero at BOTH edges (i=0 and i=H-1).
    Mimics a field with no flux crossing either boundary: only local
    compression/expansion inside a closed domain, the case mass-preserving action is
    designed for and the only case BLUR_STATUS_REPORT.md's swirl fixture exercises.
  - "open": v = V0 * i / (H-1)  -- zero at the top edge, ramps to the full magnitude V0
    right at the bottom edge (i=H-1). Mimics a diaphragm-like open boundary: real,
    nonzero velocity penetrating the domain edge.

V0 is swept across the literature-reported range of diaphragm excursion: ~15-23mm for
quiet tidal breathing and ~53-69mm for deep/forced breathing (see report references).

For each field and each of the two GroupAction.action modes ("geometric",
"mass_preserving"), we run the *unmodified* FlowDeformationOperator.forward(...,
superres=True) and record total image intensity per frame (the quantity
mass_preserving_action is supposed to keep constant). We also track, per frame, the
maximum *source* y-coordinate reachable by any output pixel under the composed
deformation field -- this exposes the actual mechanism behind any catastrophic
(complete) intensity loss, distinct from the boundary Jacobian artifact quantified in
quantify_boundary_jacobian.py.

Run from the repo root:
    python operators/lddmm/boundary_flux_report_assets/quantify_mass_preserving_flux.py
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import grid_sample

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from operators.lddmm.deform import FlowDeformationOperator, GroupAction

OUT_DIR = os.path.dirname(__file__)
torch.manual_seed(0)

H = W = 128
EXTENT = ((0.0, 450.0), (0.0, 450.0))  # project's real DEFORM_PARAMS extent (mm)
PX_PER_MM = H / 450.0

BLOCK_ROWS = (108, 128)  # bar phantom occupying the last 20 rows (near the bottom edge)
BLOCK_HU = 1000.0


def make_template():
    x = torch.zeros(1, H, W)
    x[0, BLOCK_ROWS[0] : BLOCK_ROWS[1], :] = BLOCK_HU
    return x


def make_field(kind: str, v0_mm: float) -> torch.Tensor:
    v0_px = v0_mm * PX_PER_MM
    v = torch.zeros(1, H, W, 2)
    i_idx = torch.arange(H, dtype=torch.float32)
    if kind == "closed":
        profile = torch.sin(torch.pi * i_idx / (H - 1))
    elif kind == "open":
        profile = i_idx / (H - 1)
    else:
        raise ValueError(kind)
    v[0, :, :, 1] = v0_px * profile.view(H, 1)
    return v


def run_sequence(template, v, action):
    op = FlowDeformationOperator(N=7, action=action, integration="euler", extent=EXTENT, shape=(H, W))
    seq = op(template, v, superres=True)  # list of 7 frames
    totals = [template.sum().item()] + [f.sum().item() for f in seq]
    return totals, op, seq


def reach_profile(op: FlowDeformationOperator, v: torch.Tensor):
    """Per-frame max normalized source-y coordinate reachable by ANY output pixel
    (i.e. max over the whole phi[...,1] field), and whether any sample point is
    literally out-of-bounds ([-1,1]) at any point in the sequence."""
    v_norm = op._normalize(v)
    phi_list = op.velocity_integrator(v_norm, superres=True)
    reach, oob_frac = [], []
    for phi in phi_list:
        ysrc = phi[0, :, :, 1]
        reach.append(float(ysrc.max()))
        oob_frac.append(float(((ysrc < -1) | (ysrc > 1)).float().mean()))
    return reach, oob_frac


def confirm_zero_padding_semantics():
    """Minimal, direct confirmation of what padding_mode="zeros" does for a sample
    point that is genuinely outside [-1, 1] (content that should be sourced from
    outside the template's support): grid_sample returns exactly 0 there, regardless
    of what real (nonzero) tissue might physically exist just beyond the imaged FOV.
    Contrasted with padding_mode="border", which at least propagates the edge value
    instead of injecting a fabricated zero/air value."""
    img = torch.zeros(1, 1, 8, 8)
    img[0, 0, 6:8, :] = 500.0  # bright content at the near edge
    grid = torch.zeros(1, 8, 1, 2)  # H_out=8, W_out=1
    grid[0, :, 0, 0] = 0.0  # x: center column
    grid[0, :, 0, 1] = torch.linspace(0.6, 1.4, steps=8)  # y: sweeps from inside to OOB
    out_zeros = grid_sample(img, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
    out_border = grid_sample(img, grid, align_corners=True, mode="bilinear", padding_mode="border")
    return {
        "sample_y_coords": grid[0, :, 0, 1].tolist(),
        "padding_zeros_output": out_zeros[0, 0, :, 0].tolist(),
        "padding_border_output": out_border[0, 0, :, 0].tolist(),
    }


def main():
    template = make_template()
    results = {"excursions_mm": {}, "zero_padding_confirmation": confirm_zero_padding_semantics()}

    excursions = [20.0, 60.0, 100.0]  # quiet/deep-breathing-scale (see refs), + 100mm stress test
    frame_records = {}

    for v0 in excursions:
        entry = {}
        for kind in ["closed", "open"]:
            v = make_field(kind, v0)
            sub = {}
            for action in ["geometric", "mass_preserving"]:
                totals, op, seq = run_sequence(template, v, action)
                pct = [100.0 * t / totals[0] for t in totals]
                sub[action] = {"total_intensity_pct_of_template": pct}
                if action == "geometric":
                    reach, oob = reach_profile(op, v)
                    sub["reach_profile_max_source_y"] = reach
                    sub["oob_fraction_per_frame"] = oob
                frame_records[(v0, kind, action)] = seq
            entry[kind] = sub
        results["excursions_mm"][str(v0)] = entry

    out_json = os.path.join(OUT_DIR, "mass_preserving_flux_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["excursions_mm"], indent=2))

    # --- Figure 1: total intensity trajectories, closed vs open, geometric vs mass_preserving ---
    fig, axes = plt.subplots(1, len(excursions), figsize=(6.5 * len(excursions), 4.5), sharey=False)
    if len(excursions) == 1:
        axes = [axes]
    frames_x = list(range(0, 8))
    for ax, v0 in zip(axes, excursions):
        entry = results["excursions_mm"][str(v0)]
        for kind, style in [("closed", "-"), ("open", "--")]:
            for action, color in [("geometric", "gray"), ("mass_preserving", "crimson")]:
                pct = entry[kind][action]["total_intensity_pct_of_template"]
                ax.plot(frames_x, pct, style, color=color, marker="o", markersize=3,
                         label=f"{kind}/{action}")
        ax.axhline(100, color="black", linewidth=0.7, alpha=0.5)
        ax.set_title(f"peak boundary excursion = {v0:.0f} mm")
        ax.set_xlabel("frame (N=7 euler steps)")
        ax.set_ylabel("total intensity (% of template)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "total_intensity_vs_frame.png")
    plt.savefig(fig1_path, dpi=130)
    plt.close(fig)

    # --- Figure 2: reachable-source-range contraction (the collapse mechanism) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    block_y_lo = -1 + 2 * BLOCK_ROWS[0] / (H - 1)
    for v0, style in zip(excursions, ["-", "--"]):
        entry = results["excursions_mm"][str(v0)]
        reach_open = entry["open"]["reach_profile_max_source_y"]
        reach_closed = entry["closed"]["reach_profile_max_source_y"]
        ax.plot(frames_x[1:], reach_open, style, color="crimson", marker="o", markersize=3,
                 label=f"open, {v0:.0f}mm")
        ax.plot(frames_x[1:], reach_closed, style, color="steelblue", marker="s", markersize=3,
                 label=f"closed, {v0:.0f}mm")
    ax.axhline(block_y_lo, color="black", linewidth=1.0, linestyle=":",
               label="phantom's near edge (source-y)")
    ax.set_xlabel("frame")
    ax.set_ylabel("max reachable source-y (normalized)")
    ax.set_title("Reachable source range vs. phantom location")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "reach_contraction.png")
    plt.savefig(fig2_path, dpi=130)
    plt.close(fig)

    # --- Figure 3: frame gallery for the 60mm open/mass_preserving case (visual) ---
    seq = frame_records[(60.0, "open", "mass_preserving")]
    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    axes[0].imshow(template.squeeze(), cmap="gray", vmin=0, vmax=BLOCK_HU)
    axes[0].set_title("template")
    for k, f in enumerate(seq):
        axes[k + 1].imshow(f.squeeze().detach().numpy(), cmap="gray", vmin=0, vmax=BLOCK_HU * 3)
        axes[k + 1].set_title(f"frame {k+1}")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    fig3_path = os.path.join(OUT_DIR, "frame_gallery_open_60mm.png")
    plt.savefig(fig3_path, dpi=130)
    plt.close(fig)

    print(f"\nSaved: {fig1_path}\nSaved: {fig2_path}\nSaved: {fig3_path}\nSaved: {out_json}")


if __name__ == "__main__":
    main()
