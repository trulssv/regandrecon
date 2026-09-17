"""
Quantify the progressive-blur phenomenon in FlowDeformationOperator's temporally
super-resolved deformation sequences.

Reproduces the exact setup used in test/test_deform.py and
test/test_diffeomorphic_registration.py (Shepp-Logan phantom, swirl velocity field,
N=7 integration steps, euler integration -- the project default in DEFORM_PARAMS),
then measures sharpness/information-content metrics per frame of the sequence
produced by FlowDeformationOperator.forward(template, v, superres=True).

Run from the repo root:
    python /path/to/quantify_blur.py
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from operators.lddmm.deform import FlowDeformationOperator, VelocityIntegrator
from sigpy import shepp_logan

OUT_DIR = os.path.dirname(__file__)

torch.manual_seed(0)


def get_velocity_field(shape=128, magnitude=50.0):
    """Same swirling velocity field as test/test_deform.py, vectorized."""
    coords = torch.arange(shape, dtype=torch.float32)
    ii, jj = torch.meshgrid(coords - shape / 2, coords - shape / 2, indexing="ij")
    distance = torch.sqrt(ii**2 + jj**2).clamp(min=1)
    v = torch.zeros((1, shape, shape, 2), dtype=torch.float32)
    v[0, :, :, 0] = -magnitude * jj / distance
    v[0, :, :, 1] = magnitude * ii / distance
    return v


# ---------------------------------------------------------------------------
# Sharpness / information-content metrics
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


def rms_displacement(phi: torch.Tensor, identity: torch.Tensor, shape: int) -> float:
    """RMS displacement magnitude of phi from identity, in voxels (denormalized)."""
    diff = (phi - identity).squeeze(0)  # (H, W, 2), normalized [-1,1] units
    diff_vox = diff.clone()
    diff_vox[..., 0] *= (shape - 1) / 2.0
    diff_vox[..., 1] *= (shape - 1) / 2.0
    return float(torch.sqrt((diff_vox**2).sum(-1)).mean().item())


def metrics_for(img_t: torch.Tensor) -> dict:
    img = to_numpy(img_t)
    return {
        "laplacian_var": laplacian_variance(img),
        "grad_energy": gradient_energy(img),
        "hf_energy_frac": high_freq_energy_fraction(img),
    }


def run_experiment(integration: str, N: int, shape: int = 128, magnitude: float = 50.0):
    phantom = torch.from_numpy(shepp_logan((shape, shape), dtype=np.float32)).unsqueeze(0)
    v = get_velocity_field(shape, magnitude)

    deform_op = FlowDeformationOperator(N=N, action="geometric", integration=integration, shape=(shape, shape))
    sequence = deform_op.forward(phantom, v, superres=True)  # list of N frames, 1 additional composition each

    frames = [phantom] + sequence  # frame 0 = template

    vi: VelocityIntegrator = deform_op.velocity_integrator
    identity = vi.id.unsqueeze(0) if vi.id.dim() == 3 else vi.id

    rows = []
    for k, img in enumerate(frames):
        row = {"frame": k, "n_compositions": max(k - 1, 0)}
        row.update(metrics_for(img))
        rows.append(row)

    template_metrics = rows[0]
    for row in rows:
        row["laplacian_var_pct_of_template"] = 100.0 * row["laplacian_var"] / template_metrics["laplacian_var"]
        row["grad_energy_pct_of_template"] = 100.0 * row["grad_energy"] / template_metrics["grad_energy"]
        row["hf_energy_frac_pct_of_template"] = 100.0 * row["hf_energy_frac"] / template_metrics["hf_energy_frac"]

    return frames, rows


def run_oneshot_comparison(N: int, shape: int = 128, magnitude: float = 50.0):
    """Isolate composition-interpolation blur from single-resample blur: build the FINAL
    deformation field of the euler sequence two ways -- (a) via N-1 sequential bilinear
    compositions (as FlowDeformationOperator does), and (b) directly, in one shot, by
    finely integrating the ODE (large M substeps, but composing only onto the identity
    grid analytically via direct high-resolution Euler *without* repeated bilinear
    self-composition -- instead using a single accumulated-displacement approximation).
    This estimates how much of the total blur is attributable to repeated bilinear
    composition of the *field* vs. the unavoidable single resampling of the *image*.
    """
    phantom = torch.from_numpy(shepp_logan((shape, shape), dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    v = get_velocity_field(shape, magnitude)

    deform_op = FlowDeformationOperator(N=N, action="geometric", integration="euler", shape=(shape, shape))
    v_norm = deform_op._normalize(v)
    vi = deform_op.velocity_integrator

    # (a) compounded composition (what the code actually does)
    phi_compounded = vi.euler(v_norm, N=N, superres=False)
    img_compounded = F.grid_sample(phantom, phi_compounded, align_corners=True, mode="bilinear", padding_mode="zeros")

    # (b) one-shot: same total displacement field (phi0 scaled by N, i.e. phi = id - v),
    # applied via a SINGLE bilinear resample of the template. This has the same total
    # displacement as full Euler integration would produce for a *translationally additive*
    # field, isolating the "just one interpolation" baseline blur.
    phi_oneshot = vi.id - v_norm
    img_oneshot = F.grid_sample(phantom, phi_oneshot, align_corners=True, mode="bilinear", padding_mode="zeros")

    return {
        "compounded": metrics_for(img_compounded),
        "oneshot": metrics_for(img_oneshot),
        "template": metrics_for(phantom),
    }


if __name__ == "__main__":
    results = {}

    for integration in ["euler", "scaling_and_squaring"]:
        for N in [4, 7, 10]:
            frames, rows = run_experiment(integration=integration, N=N)
            results[f"{integration}_N{N}"] = rows
            print(f"\n=== integration={integration}, N={N} ===")
            for row in rows:
                print(f"  frame {row['frame']:2d} (compositions={row['n_compositions']:2d}): "
                      f"lap_var={row['laplacian_var']:.2f} ({row['laplacian_var_pct_of_template']:5.1f}% of template)  "
                      f"grad_energy={row['grad_energy']:.5f} ({row['grad_energy_pct_of_template']:5.1f}%)  "
                      f"hf_frac={row['hf_energy_frac']:.4f} ({row['hf_energy_frac_pct_of_template']:5.1f}%)")

    print("\n=== one-shot vs compounded (N=7, euler) ===")
    comp = run_oneshot_comparison(N=7)
    print(json.dumps(comp, indent=2))

    with open(os.path.join(OUT_DIR, "blur_quantification_results.json"), "w") as f:
        json.dump({"sequences": results, "oneshot_comparison": comp}, f, indent=2)

    # Save example frames for visual inspection
    frames_euler7, _ = run_experiment(integration="euler", N=7)
    np.save(os.path.join(OUT_DIR, "frames_euler_N7.npy"), np.stack([to_numpy(f) for f in frames_euler7]))

    print("\nSaved results to", OUT_DIR)
