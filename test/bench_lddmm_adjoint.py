"""
Benchmarks the approximations of the adjoint derivative [dW|_f(v)]^* f~ of the flow deformation operator (operators/lddmm/adjoint.py) on real
4DCT time bins: accuracy against the exact (autograd) adjoint and against the converged quadrature of the continuous formula, and the time
and peak GPU memory of computing them, with and without backpropagating through them (as when training the nets that get them).

The velocity fields are realistic: v is fitted by registering a time bin f_t of a simulated study to another one f_s (consecutive bins, and
the bins of largest motion), with the operator of the model, and then scaled by a few amplitudes. Both inputs of the adjoint are tested:
    residual   f~ = W(f, v) - f_target, which makes the adjoint the gradient of 1/2 ||W(f, v) - f_target||^2
    target     f~ = f_target

Run from the repo root, e.g.
    python -m test.bench_lddmm_adjoint                     # full resolution, GPU if available
    python -m test.bench_lddmm_adjoint --downsample 2 --device cpu
Results are printed and written to --out (json).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn.functional import interpolate

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RegAndReconDataset
from operators.lddmm.deform import FlowDeformationOperator
from operators.lddmm.adjoint import adjoint_derivative, flow, smooth, spacing, warp

METHODS = ["right", "left", "midpoint", "trapezoid", "gauss2", "gauss4", "autograd"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_SIMULATED_ROOT)
    p.add_argument("--quality", default="medium")
    p.add_argument("--mode", default="val")
    p.add_argument("--study-idx", type=int, default=0)
    p.add_argument("--downsample", type=int, default=1, help="In-plane downsampling factor (the extent is unchanged).")
    p.add_argument("--device", default="auto")
    p.add_argument("--N", type=int, default=5, help="Scaling and squaring steps, as the staged config.")
    p.add_argument("--backend", default="grid_pull", choices=["grid_sample", "grid_pull"])
    p.add_argument("--fit-iters", type=int, default=80)
    p.add_argument("--amplitudes", type=float, nargs="*", default=[0.1, 0.5, 1.0, 2.0])
    p.add_argument("--sigma-mm", type=float, default=5.0, help="Width of the Gaussian K of the V-adjoint, for the comparison after smoothing.")
    p.add_argument("--timing-reps", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("test/runs/lddmm_adjoint/bench.json"))
    return p.parse_args()


def fit_velocity(deform: FlowDeformationOperator, f: torch.Tensor, target: torch.Tensor, iters: int, sigma_mm: float = 8.0, lr: float = 0.3) -> torch.Tensor:
    """A smooth v with W(f, v) ~ target: Adam on v = K p, with K a Gaussian smoothing (keeps v in the RKHS-like regime), a cosine decaying
    learning rate (in mm per step: larger ones, e.g. 2, diverge to implausible fields of tens of mm), returning the best iterate."""
    h = spacing(deform, f.shape[1:])
    p = torch.zeros(f.shape[0], 3, *f.shape[1:], device=f.device, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    best, best_p = float("inf"), p.detach().clone()
    for _ in range(iters):
        v = smooth(p, sigma_mm, h)
        loss = 0.5 * ((warp(deform, f, flow(deform, v, 1.0)) - target) ** 2).sum()
        if loss.item() < best:
            best, best_p = loss.item(), p.detach().clone()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return smooth(best_p, sigma_mm, h)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device, reps: int) -> dict:
    """Median time (s) and peak GPU memory above the baseline (MB) of fn()."""
    fn()  # warm up
    sync(device)
    times, peak = [], 0.0
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
        if device.type == "cuda":
            peak = max(peak, (torch.cuda.max_memory_allocated(device) - base) / 2**20)
    times.sort()
    return {"time_s": times[len(times) // 2], "peak_mb": peak if device.type == "cuda" else None}


def main() -> None:
    args = parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(0)

    ds = RegAndReconDataset([args.quality], args.mode, data_root=args.data_root, load_volume=True)
    item = ds[args.study_idx]
    meta = item["meta"]
    vol = item["volume"].float() / 0.019                                  # (T, D, H, W) in units of water, as the iterations
    if args.downsample > 1:
        D, H, W = vol.shape[1:]
        vol = interpolate(vol[None], size=(D, H // args.downsample, W // args.downsample), mode="trilinear", align_corners=True)[0]
    vol = vol.to(device)
    shape = tuple(vol.shape[1:])
    extent = tuple(float(e) for e in meta["extent"])
    deform = FlowDeformationOperator(N=args.N, shape=shape, extent=extent, device=device, backend=args.backend,
                                     integrator_backend="grid_sample", order=3)
    h = spacing(deform, shape)
    print(f"study {meta['study_id']}, shape {shape}, spacing {tuple(round(x, 2) for x in h)} mm, device {device}, backend {args.backend}")

    # The pair of largest motion: the bin furthest from bin 0 in image difference.
    diffs = [(vol[t] - vol[0]).norm().item() for t in range(vol.shape[0])]
    t_far = max(range(vol.shape[0]), key=lambda t: diffs[t])
    pairs = {"consecutive (0->1)": (0, 1), f"largest (0->{t_far})": (0, t_far)}

    interior = (slice(None), slice(None), slice(2, -2), slice(4, -4), slice(4, -4))
    body = None
    results: dict = {"study": meta["study_id"], "shape": shape, "device": str(device), "backend": args.backend, "accuracy": {}, "timing": {}}

    for pname, (s, t) in pairs.items():
        f, tgt = vol[s:s + 1], vol[t:t + 1]
        t0 = time.perf_counter()
        v_fit = fit_velocity(deform, f, tgt, args.fit_iters)
        with torch.no_grad():
            before = ((f - tgt) ** 2).mean().item()
            after = ((warp(deform, f, flow(deform, v_fit, 1.0)) - tgt) ** 2).mean().item()
        disp = v_fit.norm(dim=1)
        print(f"\n{pname}: fitted v in {time.perf_counter() - t0:.0f} s, max |v| {disp.max():.2f} mm, 99th pct "
              f"{disp.flatten().quantile(0.99).item() if disp.numel() < 2**24 else float('nan'):.2f} mm, mse {before:.3g} -> {after:.3g}")
        body = (f > 0.5).unsqueeze(1).expand(-1, 3, -1, -1, -1)  # soft tissue and denser, in units of water

        for amp in args.amplitudes:
            v = amp * v_fit
            with torch.no_grad():
                Wv = warp(deform, f, flow(deform, v, 1.0))
            for fname, ftilde in (("residual", Wv - tgt), ("target", tgt)):
                ref = adjoint_derivative(deform, f, v, ftilde, "autograd", create_graph=False)
                with torch.no_grad():
                    outs = {m: adjoint_derivative(deform, f, v, ftilde, m, w=Wv) for m in METHODS if m != "autograd"}
                quad = outs["gauss4"]
                row = {}
                for m, g in outs.items():
                    def rel(a, b, mask=None):
                        a, b = a[interior], b[interior]
                        if mask is not None:
                            mask = mask[interior]
                            a, b = a[mask], b[mask]
                        return ((a - b).norm() / b.norm()).item()
                    cos = torch.nn.functional.cosine_similarity(g[interior].flatten(), ref[interior].flatten(), dim=0).item()
                    gK, refK = smooth(g, args.sigma_mm, h), smooth(ref, args.sigma_mm, h)
                    # NOTE: the per-voxel error is dominated by the edges of the CT images, i.e. by high frequencies the nets' smooth
                    # velocity updates do not see. The directional ratio <G, u> / <G_exact, u> along the smoothed exact gradient u = K G_exact
                    # is the predicted change of <W(f, v), f~> for a smooth update, relative to the exact one (1 is exact).
                    u = refK / refK.norm()
                    row[m] = {"rel_err_vs_autograd": rel(g, ref), "rel_err_vs_autograd_body": rel(g, ref, body),
                              "rel_err_vs_autograd_K": rel(gK, refK), "directional_ratio": ((g * u).sum() / (ref * u).sum()).item(),
                              "rel_err_vs_quadrature": rel(g, quad), "cos_vs_autograd": cos}
                key = f"{pname} | amp {amp} | {fname}"
                results["accuracy"][key] = row
                print(f"  amp {amp:>4} {fname:>8}: " + "  ".join(
                    f"{m} {r['rel_err_vs_autograd']:.2f}/{r['directional_ratio']:.2f}/{r['rel_err_vs_quadrature']:.2f}" for m, r in row.items()))
    print(f"  (relative error vs autograd / directional ratio <G,u>/<G_exact,u> along u = K G_exact (K: sigma {args.sigma_mm} mm), 1 is exact / "
          "relative error vs gauss4 quadrature of the continuous formula; interior)")

    # ---- timing and memory, at the realistic (consecutive bins) velocity
    f, tgt = vol[0:1], vol[1:2]
    v0 = fit_velocity(deform, f, tgt, 5)
    print(f"\ntiming ({args.timing_reps} reps, median), forward only / forward + backward to (f, v, f~):")

    def bench(method: str, train: bool):
        def run():
            fr = f.clone().requires_grad_(train)
            vr = v0.clone().requires_grad_(train)
            tr = tgt.clone().requires_grad_(train)
            with torch.set_grad_enabled(train):
                Wv = warp(deform, fr, flow(deform, vr, 1.0))      # the prediction the model computes anyway
                if method == "baseline (W only)":
                    out = Wv.unsqueeze(1)
                else:
                    out = adjoint_derivative(deform, fr, vr, Wv - tr, method, w=Wv, create_graph=train if method == "autograd" else None)
                if train:
                    (out ** 2).sum().backward()
        return run

    # NOTE: the exact adjoint cannot be trained through with the grid_pull action (no double backward in torch-interpol), so it is also
    # timed with the grid_sample (trilinear) action, which has it.
    deform_gs = FlowDeformationOperator(N=args.N, shape=shape, extent=extent, device=device, backend="grid_sample", integrator_backend="grid_sample")
    deform_main = deform
    for m in ["baseline (W only)"] + METHODS + ["autograd (grid_sample action)"]:
        row = {}
        deform = deform_gs if m.endswith("(grid_sample action)") else deform_main
        method = "autograd" if m.startswith("autograd") else m
        for train in (False, True):
            try:
                row["train" if train else "forward"] = timed(bench(method if m != "baseline (W only)" else m, train), device, args.timing_reps)
            except Exception as e:  # e.g. the double backward of grid_pull
                row["train" if train else "forward"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        results["timing"][m] = row

        def fmt(r):
            if "error" in r:
                return "unsupported (" + r["error"][:60] + ")"
            return f"{r['time_s'] * 1e3:8.1f} ms" + (f" {r['peak_mb']:7.0f} MB" if r["peak_mb"] is not None else "")
        print(f"  {m:>30}: {fmt(row['forward'])} | {fmt(row['train'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
