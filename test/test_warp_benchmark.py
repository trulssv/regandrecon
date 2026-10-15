"""
Benchmarks interpolation schemes for warping 3D volumes, as candidates for the image resample in GroupAction.geometric_action, which
currently uses bilinear (trilinear) grid_sample and visibly blurs deformed volumes (see research/lddmm/BLUR_STATUS_REPORT.md).

A real volume (the first time bin of the first study of the quality loader) is warped with a synthetic, breathing-like deformation:
a smooth, mostly superior-inferior displacement that peaks at the center of the body and fades towards its edges. The deformation phi is
integrated by FlowDeformationOperator itself (with the DeformParams defaults of the model), so only the image resample differs between
the methods below:

  - grid_sample linear:  the current GroupAction.geometric_action (bilinear grid_sample, zero padding).
  - grid_pull linear:    torch-interpol's linear interpolation, which should match the above and isolates the library's overhead.
  - grid_pull cubic-z:   cubic B-spline along D only (the coarse, slice axis), linear in-plane.
  - grid_pull cubic:     cubic B-spline along all axes.

grid_pull resamples with prefilter=True, i.e. it interpolates the volume rather than smoothing it, and with bound="dct2" and
extrapolate=False, i.e. zero outside the field of view (as padding_mode="zeros"), without the edge damping that bound="zero" causes in the
spline prefilter.

For each method we measure:

  - time of the forward pass (no grad) and of a forward+backward pass with gradients w.r.t. both the volume and phi, as in training,
    and the peak memory of the latter on CUDA,
  - kept gradient energy: the gradient energy of the warped volume relative to the original, per displacement amplitude,
  - round trip RMSE: the volume is warped with phi = exp(v) and back with exp(-v). Ideally this returns the original, so the error
    measures the detail lost to interpolation. The integration error of exp(-v) o exp(v) is the same for every method. The error is only
    taken where both warps sample inside the field of view.

Outputs, in test/plots/warp_benchmark/:
  - <study_id>_warped_planes.png:    axial/coronal/sagittal mid planes of the original and of the volume warped with each method, and
                                     the difference cubic - linear.
  - <study_id>_round_trip_planes.png: the round trip with linear and cubic interpolation, and their absolute errors.
  - <study_id>_results.json:          the timings and quality metrics.

Run: pytest test/test_warp_benchmark.py -s
 or: python test/test_warp_benchmark.py [--device cuda] [--repeats 10] [--amplitudes 2 5 10 20]
"""
import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import interpol
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn.functional as F

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RegAndReconDataset, get_quality_loader
from model.hlpd import DeformParams
from operators.lddmm.deform import FlowDeformationOperator

PLOT_DIR = Path("test/plots/warp_benchmark")
QUALITY, MODE = "high", "train"
AMPLITUDES_MM = (2.0, 5.0, 10.0, 20.0)   # peak superior-inferior displacement
PLOT_AMPLITUDE_MM = 10.0
PLANES = ["axial", "coronal", "sagittal"]

# A warp maps a (B, 1, D, H, W) volume and a (B, D, H, W, 3) deformation, in grid_sample's convention (normalized to [-1, 1], channels
# (x, y, z)), to the warped (B, 1, D, H, W) volume.
Warp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _to_voxel(phi: torch.Tensor) -> torch.Tensor:
    """A deformation in grid_sample's convention (align_corners=True) in the convention of grid_pull: voxel coordinates, channels (d, h, w)."""
    sizes = phi.shape[1:-1]  # (D, H, W)
    return torch.stack([(phi[..., 2 - i] + 1) / 2 * (sizes[i] - 1) for i in range(3)], dim=-1)


def _grid_pull(order: int | list[int]) -> Warp:
    def warp(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # NOTE: a list of orders is in the order of the spatial dims of x, i.e. (D, H, W), despite grid_pull's docstring saying [W, H, D].
        return interpol.grid_pull(x, _to_voxel(phi), interpolation=order, bound="dct2", extrapolate=False, prefilter=order != 1) # type: ignore
    return warp


def _grid_sample(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(x, phi, mode="bilinear", align_corners=True, padding_mode="zeros")  # as GroupAction.geometric_action


METHODS: dict[str, Warp] = {
    "grid_sample linear": _grid_sample,
    "grid_pull linear": _grid_pull(1),
    "grid_pull cubic-z": _grid_pull([3, 1, 1]),
    "grid_pull cubic": _grid_pull(3),
}
LINEAR, CUBIC = "grid_sample linear", "grid_pull cubic"


# ---------------------------------------------------------------------------
# Data and deformation
# ---------------------------------------------------------------------------

def load_volume(quality: str = QUALITY, mode: str = MODE, data_root: Path = DEFAULT_SIMULATED_ROOT) -> tuple[torch.Tensor, dict] | None:
    """The first time bin (1, 1, D, H, W) of the first study of the quality loader and its meta data, or None if there is no study."""
    if len(RegAndReconDataset([quality], mode, data_root=data_root)) == 0:
        return None
    batch = next(iter(get_quality_loader([quality], mode, batch_size=1, shuffle=False, load_volume=True, data_root=data_root)))
    return batch["volume"][0].unsqueeze(1).float(), batch["meta"][0]


def breathing_velocity(shape: tuple[int, int, int], amplitude_mm: float) -> torch.Tensor:
    """A smooth (1, D, H, W, 3) velocity field in mm, channels (x, y, z) as FlowDeformationOperator expects: a superior-inferior (z)
    displacement of up to amplitude_mm and an anterior-posterior (y) one of 0.3 * amplitude_mm, peaking at the center of the volume."""
    D, H, W = shape
    z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in (D, H, W)], indexing="ij")
    profile = torch.exp(-(x ** 2 + y ** 2) / (2 * 0.4 ** 2)) * torch.cos(math.pi / 2 * z) ** 2
    return torch.stack([torch.zeros_like(profile), 0.3 * amplitude_mm * profile, amplitude_mm * profile], dim=-1).unsqueeze(0)


def deformation(op: FlowDeformationOperator, v: torch.Tensor) -> torch.Tensor:
    """phi = exp(v) in grid_sample's convention, integrated as in FlowDeformationOperator.forward."""
    phi = op.velocity_integrator(op._normalize(v))
    assert isinstance(phi, torch.Tensor)
    return phi


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_warp(warp: Warp, x: torch.Tensor, phi: torch.Tensor, repeats: int, backward: bool) -> dict:
    """Median and min wall time in seconds over repeats (after one warm-up), and with backward=True, the peak CUDA memory in MiB."""
    device = x.device
    x, phi = x.detach().requires_grad_(backward), phi.detach().requires_grad_(backward)
    times = []
    for i in range(repeats + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(backward):
            y = warp(x, phi)
            if backward:
                y.square().mean().backward()
        _sync(device)
        if i > 0:
            times.append(time.perf_counter() - start)
        x.grad, phi.grad = None, None
    result = {"median_s": statistics.median(times), "min_s": min(times)}
    if device.type == "cuda":
        result["peak_mem_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
    return result


def gradient_energy(x: torch.Tensor) -> float:
    """Mean squared forward difference of a (B, 1, D, H, W) volume, summed over the spatial axes (in voxel units)."""
    return sum(torch.diff(x, dim=d).square().mean().item() for d in (-3, -2, -1))


def round_trip(warp: Warp, x: torch.Tensor, phi: torch.Tensor, phi_inv: torch.Tensor) -> torch.Tensor:
    return warp(warp(x, phi), phi_inv)


def round_trip_mask(phi: torch.Tensor, phi_inv: torch.Tensor) -> torch.Tensor:
    """Where both warps of the round trip sample inside the field of view."""
    ones = torch.ones_like(phi[..., :1]).movedim(-1, 1)
    return round_trip(_grid_sample, ones, phi, phi_inv) > 1 - 1e-4


@torch.no_grad()
def quality_metrics(x: torch.Tensor, op: FlowDeformationOperator, amplitudes: tuple[float, ...]) -> dict:
    """Per method and amplitude: the kept gradient energy of the warped volume and the RMSE of the round trip, relative to the
    original's RMS, inside the round trip mask."""
    ge_x = gradient_energy(x)
    results: dict = {name: {} for name in METHODS}
    for amplitude in amplitudes:
        v = breathing_velocity(tuple(x.shape[2:]), amplitude).to(x.device)
        phi, phi_inv = deformation(op, v), deformation(op, -v)
        mask = round_trip_mask(phi, phi_inv)
        rms_x = x[mask].square().mean().sqrt().item()
        for name, warp in METHODS.items():
            err = (round_trip(warp, x, phi, phi_inv) - x)[mask]
            results[name][f"{amplitude:g}mm"] = {
                "kept_gradient_energy": gradient_energy(warp(x, phi)) / ge_x,
                "round_trip_rel_rmse": err.square().mean().sqrt().item() / rms_x,
            }
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# (label, plane, crop) per row of the plane plots. A crop is (row_lo, row_hi, col_lo, col_hi) as fractions of the plane. The zoom is on
# the base of the right lung and the diaphragm, where the superior-inferior motion and the coarse slice spacing make the blur most visible.
VIEWS: list[tuple[str, str, tuple[float, float, float, float] | None]] = [
    ("axial", "axial", None),
    ("coronal", "coronal", None),
    ("sagittal", "sagittal", None),
    ("coronal, zoom", "coronal", (0.25, 0.75, 0.12, 0.5)),
]


def _plane(x: torch.Tensor, plane: str) -> torch.Tensor:
    """The mid plane of a (1, 1, D, H, W) volume, as (rows, cols) with rows along the first remaining axis."""
    v = x[0, 0]
    axis = PLANES.index(plane)
    return v.select(axis, v.shape[axis] // 2).detach().cpu()


def _view(x: torch.Tensor, extent: tuple[float, float, float], plane: str, crop: tuple[float, float, float, float] | None
          ) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
    """The (possibly cropped) mid plane of x and its physical extent in imshow's (left, right, bottom, top) convention."""
    s = _plane(x, plane)
    rows, cols = [e for i, e in enumerate(extent) if i != PLANES.index(plane)]
    r0, r1, c0, c1 = crop if crop is not None else (0.0, 1.0, 0.0, 1.0)
    (n_rows, n_cols) = s.shape
    i0, i1, j0, j1 = round(r0 * n_rows), round(r1 * n_rows), round(c0 * n_cols), round(c1 * n_cols)
    return s[i0:i1, j0:j1], (j0 / n_cols * cols, j1 / n_cols * cols, i0 / n_rows * rows, i1 / n_rows * rows)


def plot_planes(columns: list[tuple[str, torch.Tensor, bool]], extent: tuple[float, float, float], title: str, savepath: Path) -> Path:
    """One row per view (see VIEWS), one column per (title, volume, is_difference) entry. Images share a gray scale, differences a
    symmetric one, each with one colorbar per row."""
    images = [vol for _, vol, diff in columns if not diff]
    vmin, vmax = min(v.min().item() for v in images), max(v.max().item() for v in images)
    dmax = max([v.abs().max().item() for _, v, diff in columns if diff] + [1e-12])

    fig, axes = plt.subplots(len(VIEWS), len(columns), figsize=(3.4 * len(columns) + 1.5, 3.2 * len(VIEWS)), squeeze=False, layout="constrained")
    for r, (label, plane, crop) in enumerate(VIEWS):
        ims = {}
        for c, (name, vol, diff) in enumerate(columns):
            ax = axes[r, c]
            kwargs = dict(cmap="RdBu_r", vmin=-dmax, vmax=dmax) if diff else dict(cmap="gray", vmin=vmin, vmax=vmax)
            s, ext = _view(vol, extent, plane, crop)
            ims[diff] = ax.imshow(s.numpy(), origin="lower", aspect="equal", extent=ext, interpolation="nearest", **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
        for diff, im in ims.items():
            fig.colorbar(im, ax=[axes[r, c] for c, (_, _, d) in enumerate(columns) if d == diff], shrink=0.8)
    fig.suptitle(title, fontsize=11)

    os.makedirs(savepath.parent, exist_ok=True)
    print("Saving planes to:", savepath)
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


@torch.no_grad()
def plot_comparisons(x: torch.Tensor, op: FlowDeformationOperator, extent: tuple[float, float, float], study_id: str, plot_dir: Path,
                     amplitude: float = PLOT_AMPLITUDE_MM) -> list[Path]:
    v = breathing_velocity(tuple(x.shape[2:]), amplitude).to(x.device)
    phi, phi_inv = deformation(op, v), deformation(op, -v)

    warped = {name: warp(x, phi) for name, warp in METHODS.items() if name != "grid_pull linear"}
    columns = [("original", x, False)] + [(name, y, False) for name, y in warped.items()]
    columns.append(("cubic - linear", warped[CUBIC] - warped[LINEAR], True))
    warped_path = plot_planes(columns, extent, f"{study_id}: warped with a {amplitude:g} mm breathing-like deformation [mm$^{{-1}}$]",
                              plot_dir / f"{study_id}_warped_planes.png")

    mask = round_trip_mask(phi, phi_inv)
    trips = {name: round_trip(METHODS[name], x, phi, phi_inv) for name in (LINEAR, CUBIC)}
    columns = [("original", x, False)] + [(f"round trip, {name}", y, False) for name, y in trips.items()]
    columns += [(f"error, {name}", (y - x) * mask, True) for name, y in trips.items()]
    round_trip_path = plot_planes(columns, extent, f"{study_id}: warped with exp(v) and back with exp(-v), {amplitude:g} mm [mm$^{{-1}}$]",
                                  plot_dir / f"{study_id}_round_trip_planes.png")
    return [warped_path, round_trip_path]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(device: torch.device = torch.device("cpu"), repeats: int = 5, amplitudes: tuple[float, ...] = AMPLITUDES_MM,
                  quality: str = QUALITY, mode: str = MODE, data_root: Path = DEFAULT_SIMULATED_ROOT, plot_dir: Path = PLOT_DIR) -> dict | None:
    loaded = load_volume(quality, mode, data_root)
    if loaded is None:
        return None
    x, meta = loaded
    x = x.to(device)
    shape, extent = tuple(x.shape[2:]), tuple(float(e) for e in meta["extent"])
    deform_params = DeformParams(extent=extent)
    op = FlowDeformationOperator(**asdict(deform_params), shape=shape, device=device)
    print(f"Study {meta['study_id']}: volume {shape}, extent {extent} mm, voxel {tuple(round(e / s, 2) for e, s in zip(extent, shape))} mm, "
          f"{deform_params.integration} N={deform_params.N}, device {device}")

    phi = deformation(op, breathing_velocity(shape, PLOT_AMPLITUDE_MM).to(device))
    timings = {name: {"forward": time_warp(warp, x, phi, repeats, backward=False), "forward_backward": time_warp(warp, x, phi, repeats, backward=True)}
               for name, warp in METHODS.items()}
    quality_results = quality_metrics(x, op, amplitudes)
    plots = plot_comparisons(x, op, extent, meta["study_id"], plot_dir)

    results = {"study_id": meta["study_id"], "shape": shape, "extent": extent, "device": str(device), "repeats": repeats,
               "deform_params": asdict(deform_params), "timings": timings, "quality": quality_results, "plots": [str(p) for p in plots]}
    os.makedirs(plot_dir, exist_ok=True)
    with open(plot_dir / f"{meta['study_id']}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print_results(results)
    return results


def print_results(results: dict) -> None:
    amplitudes = list(next(iter(results["quality"].values())).keys())
    header = f"{'method':<20} {'fwd [s]':>9} {'fwd+bwd [s]':>12} {'mem [MiB]':>10} | " + " ".join(f"{a + ' GE/RT':>16}" for a in amplitudes)
    print(f"\nWarp benchmark on {results['device']} (median of {results['repeats']}), kept gradient energy (GE) / round trip relative RMSE (RT):")
    print(header)
    print("-" * len(header))
    for name, timing in results["timings"].items():
        mem = timing["forward_backward"].get("peak_mem_mib")
        q = results["quality"][name]
        print(f"{name:<20} {timing['forward']['median_s']:>9.3f} {timing['forward_backward']['median_s']:>12.3f} {'-' if mem is None else f'{mem:.0f}':>10} | "
              + " ".join(f"{q[a]['kept_gradient_energy']:>8.3f}/{q[a]['round_trip_rel_rmse']:<7.4f}" for a in amplitudes))


@pytest.mark.skipif(not DEFAULT_SIMULATED_ROOT.exists(), reason=f"No simulated data in {DEFAULT_SIMULATED_ROOT}")
def test_warp_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_benchmark(device, repeats=1, amplitudes=(PLOT_AMPLITUDE_MM,))
    if results is None:
        pytest.skip(f"No simulated {QUALITY}/{MODE} studies.")
    for path in results["plots"]:
        assert Path(path).exists()

    q = {name: r[f"{PLOT_AMPLITUDE_MM:g}mm"] for name, r in results["quality"].items()}
    # grid_pull linear is the same interpolation as grid_sample, so it should lose the same detail
    assert q["grid_pull linear"]["round_trip_rel_rmse"] == pytest.approx(q[LINEAR]["round_trip_rel_rmse"], rel=0.05)
    # cubic B-splines should lose less detail than linear interpolation
    assert q[CUBIC]["round_trip_rel_rmse"] < q[LINEAR]["round_trip_rel_rmse"]
    assert q[CUBIC]["kept_gradient_energy"] > q[LINEAR]["kept_gradient_energy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=list(AMPLITUDES_MM), help="Peak displacements in mm.")
    parser.add_argument("--quality", default=QUALITY)
    parser.add_argument("--mode", default=MODE)
    args = parser.parse_args()
    if run_benchmark(torch.device(args.device), args.repeats, tuple(args.amplitudes), args.quality, args.mode) is None:
        print(f"No simulated {args.quality}/{args.mode} studies in {DEFAULT_SIMULATED_ROOT}.")
