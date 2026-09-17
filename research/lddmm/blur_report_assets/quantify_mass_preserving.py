"""
Empirically compare `GroupAction.mass_preserving_action` against `GroupAction.geometric_action`
on the same swirl-phantom fixture used by quantify_blur.py, to test the claim (BLUR_STATUS_REPORT.md,
point C.6) that mass-preserving is "worth wiring up on its own merits" for CT intensity/quantity
conservation, and to check -- empirically, not by inference -- whether it also affects the
composition-blur trend quantified in section B.

Reproduces the exact setup used in test/test_deform.py and test/test_diffeomorphic_registration.py
(Shepp-Logan phantom, swirl velocity field, N=7 integration steps, euler integration -- the project
default in DEFORM_PARAMS), then for every frame of the temporally super-resolved deformation-field
sequence applies BOTH `geometric_action` and `mass_preserving_action` to the same phi and compares:

1. The three sharpness metrics from quantify_blur.py (gradient energy, Laplacian variance, HF FFT
   energy fraction), full-frame and with a boundary ring excluded (see point 3 below).
2. Total image intensity (sum over pixels) relative to the template, for both actions.
3. Whether `GroupAction.jacobian_determinant` goes negative/positive (folding) anywhere in the
   sequence, and what fraction of pixels, at which frames -- including a direct check of the
   determinant's *sign convention* on non-deformed/known test fields, and of `torch.roll`'s
   circular-wraparound effect at the domain boundary, since both affect how "folding" must be read.

Run from the repo root:
    python operators/lddmm/blur_report_assets/quantify_mass_preserving.py
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from operators.lddmm.deform import GroupAction, VelocityIntegrator, FlowDeformationOperator
from sigpy import shepp_logan

OUT_DIR = os.path.dirname(__file__)

torch.manual_seed(0)


def get_velocity_field(shape=128, magnitude=50.0):
    """Same swirling velocity field as test/test_deform.py / quantify_blur.py, vectorized."""
    coords = torch.arange(shape, dtype=torch.float32)
    ii, jj = torch.meshgrid(coords - shape / 2, coords - shape / 2, indexing="ij")
    distance = torch.sqrt(ii**2 + jj**2).clamp(min=1)
    v = torch.zeros((1, shape, shape, 2), dtype=torch.float32)
    v[0, :, :, 0] = -magnitude * jj / distance
    v[0, :, :, 1] = magnitude * ii / distance
    return v


# ---------------------------------------------------------------------------
# Sharpness / information-content metrics -- identical definitions to quantify_blur.py
# ---------------------------------------------------------------------------

def to_numpy(img: torch.Tensor) -> np.ndarray:
    return img.squeeze().detach().cpu().numpy()


def laplacian_variance(img: np.ndarray) -> float:
    """Classic focus measure: variance of the Laplacian. Lower => blurrier."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d
    lap = convolve2d(img, k, mode="valid")
    return float(lap.var())


def gradient_energy(img: np.ndarray) -> float:
    """Mean squared gradient magnitude (Tenengrad-style sharpness)."""
    gy, gx = np.gradient(img)
    return float(np.mean(gx**2 + gy**2))


def high_freq_energy_fraction(img: np.ndarray, cutoff_frac: float = 0.25) -> float:
    """Fraction of spectral energy lying above `cutoff_frac` of the Nyquist radius."""
    F2 = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F2) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = min(cy, cx)
    mask_hf = r > (cutoff_frac * r_max)
    total = power.sum()
    return float(power[mask_hf].sum() / total) if total > 0 else 0.0


def metrics_for(img: np.ndarray) -> dict:
    return {
        "laplacian_var": laplacian_variance(img),
        "grad_energy": gradient_energy(img),
        "hf_energy_frac": high_freq_energy_fraction(img),
    }


def crop_interior(arr: np.ndarray, exclude: int) -> np.ndarray:
    """Drop an `exclude`-pixel ring from each edge, to separate genuine field behaviour from the
    torch.roll circular-wraparound artifact at the domain boundary (see jacobian_determinant, which
    differentiates via phi.roll(shifts=+-1); the outermost ring wraps to the opposite edge of the
    grid, producing a spurious finite-difference spike unrelated to the actual deformation)."""
    if exclude <= 0:
        return arr
    return arr[exclude:-exclude, exclude:-exclude]


# ---------------------------------------------------------------------------
# 0. Sanity checks on jacobian_determinant itself: calibration and sign convention.
#    These are cheap, targeted checks (not part of the main frame-by-frame sweep) that determine
#    how the fold-fraction and mass-conservation numbers below must be interpreted.
# ---------------------------------------------------------------------------

def check_jacobian_determinant_baseline(shape: int = 128) -> dict:
    """What does jacobian_determinant report for known, hand-constructed fields?

    - Identity field (phi = id, i.e. zero deformation): the true Jacobian determinant of an
      identity map is exactly 1 everywhere. If jacobian_determinant returns something else at
      zero deformation, that is a calibration bug independent of anything the swirl fixture does.
    - Uniform isotropic scaling phi = c * id (c > 0): orientation-preserving for any c > 0, so the
      determinant's *sign* should match the identity-field sign for every c > 0 tested.
    - A single-axis mirror flip (phi_x -> -phi_x, phi_y unchanged): this is the textbook
      orientation-*reversing* map, so its determinant sign must be the opposite of the identity
      field's sign. This tells us which sign this codebase's convention actually uses for
      "folded" -- needed before any "det < 0 => folding" claim can be trusted.
    """
    ga = GroupAction(action="geometric")
    vi = VelocityIntegrator(N=7, integration="euler", shape=(shape, shape))
    idn = vi.id.unsqueeze(0)

    det_identity = ga.jacobian_determinant(idn)
    interior = crop_interior(det_identity[0].numpy(), 3)

    scale_signs = {}
    for c in (0.5, 1.0, 2.0):
        det_c = ga.jacobian_determinant(c * idn)
        scale_signs[str(c)] = float(crop_interior(det_c[0].numpy(), 3).mean())

    mirror = idn.clone()
    mirror[..., 0] = -mirror[..., 0]
    det_mirror = ga.jacobian_determinant(mirror)
    mirror_interior_mean = float(crop_interior(det_mirror[0].numpy(), 3).mean())

    return {
        "identity_det_interior_mean": float(interior.mean()),
        "identity_det_interior_min": float(interior.min()),
        "identity_det_interior_max": float(interior.max()),
        "identity_det_full_min": float(det_identity.min()),
        "identity_det_full_max": float(det_identity.max()),
        "note_full_min_max_include_boundary_roll_wraparound": True,
        "uniform_scale_det_interior_mean_by_c": scale_signs,
        "mirror_flip_det_interior_mean": mirror_interior_mean,
        "fold_sign_convention": (
            "negative (matches identity)" if mirror_interior_mean * interior.mean() < 0
            else "ambiguous/unexpected"
        ),
    }


# ---------------------------------------------------------------------------
# 1. Main per-frame comparison: geometric vs mass-preserving on the same phi_k sequence.
# ---------------------------------------------------------------------------

def run_action_comparison(integration: str, N: int, shape: int = 128, magnitude: float = 50.0,
                           boundary_exclude: int = 3):
    phantom = torch.from_numpy(shepp_logan((shape, shape), dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    v = get_velocity_field(shape, magnitude)

    # Build phi once via a FlowDeformationOperator's own integrator/normalization (matches the
    # production path exactly), then apply BOTH group actions to the identical phi_k -- this
    # guarantees geometric and mass-preserving are compared on exactly the same field per frame,
    # not on two independently-integrated (but nominally identical) sequences.
    deform_op = FlowDeformationOperator(N=N, action="geometric", integration=integration, shape=(shape, shape))
    v_norm = deform_op._normalize(v)
    vi: VelocityIntegrator = deform_op.velocity_integrator
    ga: GroupAction = deform_op.group_action  # action="geometric", but both action methods are callable directly

    if integration == "euler":
        phi_list = vi.euler(v_norm, N=N, superres=True)
    else:
        phi_list = vi.scale_and_square(v_norm, N=N, superres=True)

    template_np = to_numpy(phantom)
    template_metrics = metrics_for(template_np)
    template_metrics_interior = metrics_for(crop_interior(template_np, boundary_exclude))
    template_sum = float(phantom.sum().item())

    rows = []
    for k, phi in enumerate(phi_list):
        img_geo_t = ga.geometric_action(phantom, phi)
        img_mp_t = ga.mass_preserving_action(phantom, phi)
        det_signed = ga.jacobian_determinant(phi)  # NOTE: signed, i.e. before mass_preserving_action's abs()

        img_geo = to_numpy(img_geo_t)
        img_mp = to_numpy(img_mp_t)

        det_full = det_signed[0].numpy()
        det_interior = crop_interior(det_full, boundary_exclude)

        # Fold fraction: per check_jacobian_determinant_baseline, the non-folded/identity sign for
        # this codebase's determinant convention is NEGATIVE, so a flip to POSITIVE indicates folding,
        # not the reverse. See section D of BLUR_STATUS_REPORT.md before relying on this.
        fold_frac_full = float((det_full > 0).mean())
        fold_frac_interior = float((det_interior > 0).mean())

        outside_domain_frac = float((phi.abs() > 1).any(dim=-1).float().mean().item())

        row = {
            "frame": k + 1,
            "n_compositions": k,
            "geo": metrics_for(img_geo),
            "mp": metrics_for(img_mp),
            "geo_interior": metrics_for(crop_interior(img_geo, boundary_exclude)),
            "mp_interior": metrics_for(crop_interior(img_mp, boundary_exclude)),
            "geo_sum": float(img_geo_t.sum().item()),
            "mp_sum": float(img_mp_t.sum().item()),
            "geo_sum_pct_of_template": 100.0 * float(img_geo_t.sum().item()) / template_sum,
            "mp_sum_pct_of_template": 100.0 * float(img_mp_t.sum().item()) / template_sum,
            "det_interior_min": float(det_interior.min()),
            "det_interior_max": float(det_interior.max()),
            "det_interior_mean": float(det_interior.mean()),
            "fold_frac_full_domain": fold_frac_full,
            "fold_frac_interior": fold_frac_interior,
            "frac_phi_sampling_outside_domain": outside_domain_frac,
        }
        for key in ("laplacian_var", "grad_energy", "hf_energy_frac"):
            row[f"geo_{key}_pct_of_template"] = 100.0 * row["geo"][key] / template_metrics[key]
            row[f"mp_{key}_pct_of_template"] = 100.0 * row["mp"][key] / template_metrics[key]
            row[f"geo_interior_{key}_pct_of_template"] = 100.0 * row["geo_interior"][key] / template_metrics_interior[key]
            row[f"mp_interior_{key}_pct_of_template"] = 100.0 * row["mp_interior"][key] / template_metrics_interior[key]
        rows.append(row)

    return rows, template_metrics, template_sum


if __name__ == "__main__":
    results = {}

    print("=== 0. jacobian_determinant baseline/sign-convention checks ===")
    baseline = check_jacobian_determinant_baseline()
    print(json.dumps(baseline, indent=2))
    results["jacobian_determinant_baseline_checks"] = baseline

    print("\n=== 1. Primary fixture: euler, N=7, magnitude=50 (project DEFORM_PARAMS) ===")
    rows_primary, template_metrics, template_sum = run_action_comparison(integration="euler", N=7, magnitude=50.0)
    results["primary_euler_N7_magnitude50"] = {
        "template_metrics": template_metrics,
        "template_sum": template_sum,
        "frames": rows_primary,
    }
    for row in rows_primary:
        print(f"  frame {row['frame']}: geo_sum={row['geo_sum_pct_of_template']:6.1f}%  "
              f"mp_sum={row['mp_sum_pct_of_template']:7.1f}%  "
              f"fold_frac(interior)={row['fold_frac_interior']:.4f}  "
              f"frac_outside_domain={row['frac_phi_sampling_outside_domain']:.4f}")

    print("\n=== 2. Cheap sweep: other integration/N combos (same magnitude=50) ===")
    for integration in ["euler", "scaling_and_squaring"]:
        for N in [4, 10]:
            rows, t_metrics, t_sum = run_action_comparison(integration=integration, N=N)
            results[f"{integration}_N{N}_magnitude50"] = {
                "template_metrics": t_metrics, "template_sum": t_sum, "frames": rows,
            }
            last = rows[-1]
            print(f"  {integration} N={N}: last-frame geo_sum={last['geo_sum_pct_of_template']:.1f}%  "
                  f"mp_sum={last['mp_sum_pct_of_template']:.1f}%  fold_frac(interior)={last['fold_frac_interior']:.4f}")

    print("\n=== 3. Low-magnitude supplementary check (magnitude=5, euler, N=7) ===")
    print("    Purpose: at magnitude=50 (the stress-test fixture used above and in section B), a large")
    print("    fraction of phi samples land outside the domain (see frac_phi_sampling_outside_domain above),")
    print("    so intensity is genuinely lost from the finite grid regardless of any Jacobian correction --")
    print("    that is NOT the scenario mass-preserving is meant to address. This run uses a much smaller")
    print("    swirl magnitude to check the intensity-conservation claim in a regime closer to the assumed")
    print("    'closed boundary, no folding' case.")
    rows_lowmag, t_metrics_lowmag, t_sum_lowmag = run_action_comparison(integration="euler", N=7, magnitude=5.0)
    results["low_magnitude_euler_N7_magnitude5"] = {
        "template_metrics": t_metrics_lowmag, "template_sum": t_sum_lowmag, "frames": rows_lowmag,
    }
    for row in rows_lowmag:
        print(f"  frame {row['frame']}: geo_sum={row['geo_sum_pct_of_template']:6.1f}%  "
              f"mp_sum={row['mp_sum_pct_of_template']:7.1f}%  "
              f"fold_frac(interior)={row['fold_frac_interior']:.4f}  "
              f"frac_outside_domain={row['frac_phi_sampling_outside_domain']:.4f}")

    with open(os.path.join(OUT_DIR, "mass_preserving_quantification_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to", os.path.join(OUT_DIR, "mass_preserving_quantification_results.json"))

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames_x = [r["frame"] for r in rows_primary]

    # Mass conservation: geo vs mp, primary (magnitude=50) vs low-magnitude (magnitude=5) fixture.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].axhline(100.0, color="gray", linestyle="--", linewidth=1, label="template (100%)")
    axes[0].plot(frames_x, [r["geo_sum_pct_of_template"] for r in rows_primary], "o-", label="geometric_action")
    axes[0].plot(frames_x, [r["mp_sum_pct_of_template"] for r in rows_primary], "s-", label="mass_preserving_action")
    axes[0].set_title("Total intensity vs. frame\n(magnitude=50, euler, N=7 -- primary fixture)")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("% of template total intensity")
    axes[0].legend()

    axes[1].axhline(100.0, color="gray", linestyle="--", linewidth=1, label="template (100%)")
    axes[1].plot(frames_x, [r["geo_sum_pct_of_template"] for r in rows_lowmag], "o-", label="geometric_action")
    axes[1].plot(frames_x, [r["mp_sum_pct_of_template"] for r in rows_lowmag], "s-", label="mass_preserving_action")
    axes[1].set_title("Total intensity vs. frame\n(magnitude=5, euler, N=7 -- low-leakage supplementary check)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("% of template total intensity")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mass_conservation_curves.png"), dpi=130)
    plt.close(fig)

    # Sharpness metric trend: geometric vs mass-preserving, interior-cropped (boundary ring excluded).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_keys = ["grad_energy", "laplacian_var", "hf_energy_frac"]
    titles = ["Gradient energy (interior, % of template)", "Laplacian variance (interior, % of template)",
              "HF FFT energy fraction (interior, % of template)"]
    for ax, key, title in zip(axes, metric_keys, titles):
        ax.axhline(100.0, color="gray", linestyle="--", linewidth=1)
        ax.plot(frames_x, [r[f"geo_interior_{key}_pct_of_template"] for r in rows_primary], "o-", label="geometric")
        ax.plot(frames_x, [r[f"mp_interior_{key}_pct_of_template"] for r in rows_primary], "s-", label="mass-preserving")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("frame")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mass_preserving_sharpness_curves.png"), dpi=130)
    plt.close(fig)

    # Frame gallery: geometric vs mass-preserving side by side, primary fixture, a few frames.
    phantom = torch.from_numpy(shepp_logan((128, 128), dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    v = get_velocity_field(128, 50.0)
    deform_op = FlowDeformationOperator(N=7, action="geometric", integration="euler", shape=(128, 128))
    v_norm = deform_op._normalize(v)
    vi = deform_op.velocity_integrator
    ga = deform_op.group_action
    phi_list = vi.euler(v_norm, N=7, superres=True)

    show_frames = [0, 2, 6]  # frame indices into phi_list (0-based): 1st, 3rd, 7th (last) output frame
    fig, axes = plt.subplots(2, len(show_frames) + 1, figsize=(3.2 * (len(show_frames) + 1), 6.4))
    axes[0, 0].imshow(to_numpy(phantom), cmap="gray")
    axes[0, 0].set_title("template")
    axes[1, 0].imshow(to_numpy(phantom), cmap="gray")
    axes[1, 0].set_title("template")
    for col, idx in enumerate(show_frames, start=1):
        phi = phi_list[idx]
        img_geo = to_numpy(ga.geometric_action(phantom, phi))
        img_mp = to_numpy(ga.mass_preserving_action(phantom, phi))
        axes[0, col].imshow(img_geo, cmap="gray")
        axes[0, col].set_title(f"geometric, frame {idx + 1}")
        # mass-preserving values are dominated by extreme outliers (see report); clip for display only.
        vmax = np.percentile(np.abs(img_mp), 99.0)
        axes[1, col].imshow(img_mp, cmap="gray", vmin=0, vmax=vmax)
        axes[1, col].set_title(f"mass-preserving, frame {idx + 1}\n(clipped to 99th pct for display)")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mass_preserving_frame_gallery.png"), dpi=130)
    plt.close(fig)

    print("Saved plots to", OUT_DIR)
