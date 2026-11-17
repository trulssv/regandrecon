"""
Compares registration (motion) modules of the HLPD model in the full model, as in the motion stage of the staged training: the primal and
dual nets come from a checkpoint (by default best_recon.pt of the latest staged run, whose motion nets are still zero) and are frozen, and
only the new motion nets are trained, with the loss of the checkpoint's config, on a single training study (an overfit test, not a
training run). For each variant, it records the time and peak GPU memory of a training step, the number of parameters, and the learning
curves of the image domain error of the predictions w_t = W(f_{t-1}, v_{t-1}) (warp_mse) and of the coupling loss, and at the end the
warp_mse on a validation study.

Variants are given as <net>[+lddmm[:<method>[:<input>]]], e.g.
    conv                  the current motion nets (ConvStack, inputs v_prev, v_{t-1}, f_t, f_{t+1}, w_{t+1})
    unet                  a small U-net with the same inputs (model/motion.py)
    conv+lddmm            the LDDMM-styled inputs v_prev, v_{t-1}, [dW|_{f_t}(v_prev)]^* (W(f_t, v_prev) - f_{t+1})  (right rule)
    unet+lddmm:midpoint:target

Run from the repo root (on a free GPU), e.g.
    python -m test.compare_registration_modules --steps 300
    python -m test.compare_registration_modules --variants conv unet --steps 20 --benchmark-only
Results go to <out-dir>/<timestamp>/ (results.json, curves.png).
"""
import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RegAndReconDataset, collate_time_list
from model.build import HLPDConfig, LossParams, build_hlpd, configure_cuda, select_gpu, set_geometry, simulation_mu_water, simulation_ray_trafo_params
from model.hlpd import ConvStackParams
from operators.lddmm.adjoint import adjoint_derivative
from model.motion import LDDMMParams, UNetParams, convert_motion, count_parameters
from train.stages import merge_loss_config, parse_stages

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("/media/trulssv/LDDMM/checkpoints/20261111-131756_hlpd_train/best_recon.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint with the primal and dual nets, and the config.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_SIMULATED_ROOT)
    p.add_argument("--study", type=int, default=0, help="Index into the checkpoint run's training studies.")
    p.add_argument("--val-study", type=int, default=0, help="Index into the checkpoint run's validation studies, for the final evaluation.")
    p.add_argument("--variants", nargs="*", default=["conv", "unet", "conv+lddmm", "unet+lddmm"])
    p.add_argument("--unet-channels", type=int, nargs="*", default=[16, 32, 64])
    p.add_argument("--adjoint-scale", type=float, default=None, help="Scale a of the adjoint input, by default 1 / its RMS in the body at "
                   "zero motion between the ground truth bins 0 and 1.")
    p.add_argument("--sigma-mm", type=float, default=0.0, help="Width of the Gaussian K of the V-adjoint (0: the L^2 adjoint).")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=None, help="By default the learning rate of the checkpoint's optimizer config.")
    p.add_argument("--benchmark-only", action="store_true", help="Only time the steps (--steps of them), no evaluation.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("test/runs/registration_modules"))
    return p.parse_args()


def parse_variant(spec: str, adjoint_scale: float, sigma_mm: float) -> tuple[str, LDDMMParams | None]:
    net, _, rest = spec.partition("+")
    if not rest:
        return net, None
    parts = rest.split(":")
    assert parts[0] == "lddmm", f"Unknown variant {spec}."
    method = parts[1] if len(parts) > 1 else "right"
    adjoint_input = parts[2] if len(parts) > 2 else "residual"
    return net, LDDMMParams(method=method, adjoint_input=adjoint_input, adjoint_scale=adjoint_scale, sigma_mm=sigma_mm)


def load(data_root: Path, study_id: str, mode: str, quality: str, device: torch.device):
    ds = RegAndReconDataset([quality], mode, data_root=data_root, load_volume=True, study_ids=[study_id])
    batch = collate_time_list([ds[0]])
    return [g.to(device) for g in batch["sinogram"]], [x.to(device) for x in batch["volume"]], batch["meta"][0]


def main() -> None:
    args = parse_args()
    configure_cuda()
    device = (select_gpu() if torch.cuda.is_available() else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    logger.info("Running on %s.", device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_config = HLPDConfig.from_dict(cfg)
    # NOTE: the checkpoint stores the parsed stages, with their start epochs, which parse_stages sets itself.
    train_cfg = {**cfg["training"], "stages": [{k: v for k, v in st.items() if k != "start"} for st in cfg["training"].get("stages", [])]}
    stages = parse_stages(train_cfg if train_cfg["stages"] else cfg["training"])
    motion_stage = next((s for s in stages if "motion" in s.train and set(s.train) == {"motion"}), None)
    loss_params = LossParams.from_dict(merge_loss_config(cfg["loss"], motion_stage.loss if motion_stage is not None else {}))
    lr = args.lr if args.lr is not None else (motion_stage.lr if motion_stage is not None and motion_stage.lr else cfg["optimizer"]["params"]["lr"])
    quality = cfg["data"]["train_qualities"][0]
    ids = cfg["data"]["study_ids"]

    g, x_true, meta = load(args.data_root, ids["train"][args.study], "train", quality, device)
    g_val, x_val, meta_val = load(args.data_root, ids["val"][args.val_study], "val", quality, device)
    T, shape = len(g), tuple(x_true[0].shape[1:])
    y = g if loss_params.domain == "projection" else x_true
    y_val = g_val if loss_params.domain == "projection" else x_val
    logger.info("Train study %s, val study %s, shape %s, %d bins, loss domain %s, lr %g.", meta["study_id"], meta_val["study_id"], shape, T,
                loss_params.domain, lr)

    model, loss_fn = build_hlpd(model_config, simulation_ray_trafo_params(args.data_root), T=T, shape=shape, device=device,
                                mu_water=simulation_mu_water(args.data_root))
    loss_fn.set_params(loss_params)
    set_geometry(model, loss_fn, meta)
    state = ckpt["model"]
    del ckpt

    # The scale of the adjoint input: at zero motion between the ground truth bins 0 and 1, in the units of the iterations.
    if args.adjoint_scale is None:
        with torch.no_grad():
            f0, f1 = model.units.images([x_true[0], x_true[1]])
            adj = adjoint_derivative(model.deform, f0, torch.zeros(1, 3, *shape, device=device), f0 - f1, "right")
            body = (f0 > 0.5 * f0.amax()).unsqueeze(1).expand_as(adj) if model.normalization.intensity == "physical" else (f0 > 0.5).unsqueeze(1).expand_as(adj)
            rms = adj[body].square().mean().sqrt().item()
        args.adjoint_scale = 1.0 / rms
        logger.info("Adjoint RMS in the body %.3g, scale a = %.3g.", rms, args.adjoint_scale)

    zero_warp_mse = torch.stack([(x_true[(t - 1) % T] - x_true[t]).square().mean() for t in range(T)]).mean().item()
    results: dict = {"train_study": meta["study_id"], "val_study": meta_val["study_id"], "checkpoint": str(args.checkpoint),
                     "steps": args.steps, "lr": lr, "adjoint_scale": args.adjoint_scale, "sigma_mm": args.sigma_mm,
                     "zero_motion_warp_mse": zero_warp_mse, "variants": {}}
    logger.info("warp_mse of zero motion (w_t = x_{t-1}): %.4g", zero_warp_mse)
    out = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    for spec in args.variants:
        kind, lddmm = parse_variant(spec, args.adjoint_scale, args.sigma_mm)
        # NOTE: back to the checkpoint's motion nets (ConvStacks) first, as a previous variant replaced them.
        convert_motion(model, "conv", None, conv_stack=ConvStackParams(**cfg["conv_stack"]))
        model.load_state_dict(state)
        set_geometry(model, loss_fn, meta)
        convert_motion(model, kind, lddmm, conv_stack=ConvStackParams(**cfg["conv_stack"]), unet=UNetParams(channels=args.unet_channels), seed=args.seed)
        model.set_trainable(["motion"])
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        opt = getattr(torch.optim, cfg["optimizer"]["name"])(params, **{**cfg["optimizer"].get("params", {}), "lr": lr})
        logger.info("=== %s: %d trainable parameters (%d per motion net)", spec, n_params,
                    count_parameters(model.unrolled_model.hlpd_model_list[0].motion_net[0]))

        curve, times, peaks = [], [], []
        for step in range(args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            w, f, v = model(g)
            loss = loss_fn(w, f, v, y)
            loss.total_loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peaks.append(torch.cuda.max_memory_allocated(device) / 2**30)
            times.append(time.perf_counter() - t0)
            with torch.no_grad():
                warp_mse = torch.stack([(wt - xt).square().mean() for wt, xt in zip(w, x_true)]).mean().item()
            curve.append({"step": step, "total_loss": loss.total_loss.item(), "coupling_loss": loss.coupling_loss.item(), "warp_mse": warp_mse})
            if step % args.log_every == 0 or step == args.steps - 1:
                logger.info("%s step %d: loss %.4g, coupling %.4g, warp_mse %.4g, %.2f s, peak %.1f GB", spec, step, curve[-1]["total_loss"],
                            curve[-1]["coupling_loss"], warp_mse, times[-1], peaks[-1] if peaks else float("nan"))
            del w, f, v, loss

        row = {"trainable_parameters": n_params, "step_time_s": statistics.median(times[1:] or times),
               "peak_memory_gb": max(peaks) if peaks else None, "curve": curve}
        if not args.benchmark_only:
            with torch.no_grad():
                set_geometry(model, loss_fn, meta_val)
                w, f, v = model(g_val)
                row["val_warp_mse"] = torch.stack([(wt - xt).square().mean() for wt, xt in zip(w, x_val)]).mean().item()
                row["val_zero_motion_warp_mse"] = torch.stack([(x_val[(t - 1) % T] - x_val[t]).square().mean() for t in range(T)]).mean().item()
                row["val_total_loss"] = loss_fn(w, f, v, y_val).total_loss.item()
                del w, f, v
            logger.info("%s: val warp_mse %.4g (zero motion %.4g)", spec, row["val_warp_mse"], row["val_zero_motion_warp_mse"])
        results["variants"][spec] = row
        (out / "results.json").write_text(json.dumps(results, indent=1))
        del opt, params
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Summary (median step time, peak memory, final train warp_mse, val warp_mse):")
    for spec, row in results["variants"].items():
        logger.info("  %-28s %6.2f s  %5s GB  %.4g  %s", spec, row["step_time_s"], f"{row['peak_memory_gb']:.1f}" if row["peak_memory_gb"] else "-",
                    row["curve"][-1]["warp_mse"], f"{row['val_warp_mse']:.4g}" if "val_warp_mse" in row else "-")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for spec, row in results["variants"].items():
            steps = [c["step"] for c in row["curve"]]
            axes[0].plot(steps, [c["warp_mse"] for c in row["curve"]], label=spec)
            axes[1].plot(steps, [c["coupling_loss"] for c in row["curve"]], label=spec)
        axes[0].axhline(zero_warp_mse, color="gray", ls="--", lw=1, label="zero motion")
        for ax, title in zip(axes, ["warp_mse (train study, vs ground truth)", "coupling loss"]):
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title(title); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "curves.png", dpi=120)
    except Exception as e:  # plotting is optional
        logger.warning("No plot: %s", e)
    logger.info("Results in %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
