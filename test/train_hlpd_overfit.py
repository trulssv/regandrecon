"""
Overfits an HLPDModel to a single simulated study, as a check that the model and its loss can converge at all.

The model reconstructs the time bins f_t, the cyclic motion v_t from time bin t to t+1 mod T, and the predictions w_t = W(f_{t-1}, v_{t-1})
of each time bin by the motion from the sinogram g, and is trained with the joint reconstruction and registration loss (model/hlpd_loss.py):
    --domain image       supervised:    sum_t ||x_t - f_t||^2      + lambda_c ||w_t - f_t||^2 + lambda ||L v_t||^2, with x_t the ground truth volumes
    --domain projection  unsupervised:  sum_t ||g_t - T_t f_t||^2  + lambda_c ||w_t - f_t||^2 + lambda ||L v_t||^2, with T_t the ray transform of time bin t
The image domain errors of the reconstructions and of their predictions are logged in both cases, so that convergence can be compared.

Uses the full resolution study and its stored sinogram, so the ray transform is built from the config the data was simulated with.
Per-step losses, wall time and peak GPU memory are logged to <out-dir>/metrics/<run_id>.jsonl (see tracking/run.py). With
--profile, a few steps are additionally recorded with torch.profiler, and the time per module (dual/primal/motion nets, ray
transforms, deformation, ...) is printed and written to <out-dir>/profile/.

Run from the repo root, e.g.
    python -m test.train_hlpd_overfit --domain projection --steps 500 --profile
"""
import argparse
import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RegAndReconDataset, collate_time_list
from model.build import HLPDConfig, LossParams, build_hlpd, configure_cuda, set_geometry, simulation_mu_water, simulation_ray_trafo_params
from model.hlpd import ConvStackParams, DeformParams, HLPDBlockParams, HLPDUnrolledParams, NormalizationParams
from tracking.profiling import StepStats, make_profiler, module_time_table
from tracking.run import start_run
from tracking.timeline import StepTimeline, region
from tracking.training_plots import plot_gpu_memory, plot_losses, plot_prediction_gif, plot_step_times

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    parser.add_argument("--data-root", type=Path, default=DEFAULT_SIMULATED_ROOT)
    parser.add_argument("--quality", default="high")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--study-idx", type=int, default=0)
    # model
    parser.add_argument("--num-iters", type=int, default=5, help="Number of unrolled HLPD iterations.")
    parser.add_argument("--hidden-channels", type=int, nargs="*", default=[32, 32], help="Hidden channels of each ConvStack, must be divisible by 4 (GroupNorm).")
    parser.add_argument("--recurrent", action=argparse.BooleanOptionalAction, default=True, help="Recurrent in time. The non-recurrent model stacks all time bins as channels, which is very memory hungry at full resolution.")
    parser.add_argument("--shared-time", action=argparse.BooleanOptionalAction, default=True, help="Recurrent mode: share the nets across time bins.")
    parser.add_argument("--shared-iters", action=argparse.BooleanOptionalAction, default=False, help="Share the HLPD block across iterations.")
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--integration-steps", type=int, default=7)
    parser.add_argument("--normalize-operator", action=argparse.BooleanOptionalAction, default=True, help="Use T_t / ||T_t|| in the HLPD iterations and scale the data g_t accordingly (see model.hlpd.NormalizedRayTransform).")
    parser.add_argument("--intensity", choices=["physical", "water"], default="water", help="Units of the images in the HLPD iterations, see model.hlpd.NormalizationParams. The output is in mm^-1 in both modes.")
    parser.add_argument("--fbp-init", action=argparse.BooleanOptionalAction, default=True, help="Initialize f with the FBP and h with the data, instead of zeros.")
    # loss and optimization
    parser.add_argument("--domain", choices=["image", "projection"], default="image")
    parser.add_argument("--lambda-coupling", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0)
    parser.add_argument("--helmholtz", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("ALPHA", "GAMMA", "BETA"))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=500)
    # execution
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="Record a few steps with torch.profiler and report the time per module.")
    parser.add_argument("--profile-steps", type=int, nargs=3, default=[1, 1, 3], metavar=("WAIT", "WARMUP", "ACTIVE"))
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Write the live plots (see tracking/training_plots.py) to <plot-dir>/<run_id>/.")
    parser.add_argument("--plot-every", type=int, default=25, help="Record the GPU memory and time of the key steps of a step and rewrite the memory, timing and loss plots every this many steps.")
    parser.add_argument("--gif-every", type=int, default=100, help="Rewrite the prediction GIF every this many steps.")
    parser.add_argument("--plot-dir", type=Path, default=Path("test/plots"))
    parser.add_argument("--out-dir", type=Path, default=Path("test/runs/hlpd_overfit"))
    return parser.parse_args()


def load_study(data_root: Path, quality: str, mode: str, idx: int, device: torch.device, load_fbp: bool = False
               ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, dict]:
    """The sinogram, ground truth volume and (with load_fbp) stored FBP of one study, as lists over time of (1, ...) tensors, and the
    study's meta data."""
    dataset = RegAndReconDataset(qualities=[quality], mode=mode, data_root=data_root, load_volume=True, load_fbp=load_fbp)
    if len(dataset) == 0:
        raise RuntimeError(f"No finished '{mode}' studies of quality '{quality}' under {data_root}.")
    batch = collate_time_list([dataset[idx]])
    g = [gt.to(device) for gt in batch["sinogram"]]       # T x (1, nAngles, nCol, nRow)
    x = [xt.to(device) for xt in batch["volume"]]         # T x (1, D, H, W)
    fbp = [ft.to(device) for ft in batch["fbp"]] if load_fbp else None   # T x (1, D, H, W)
    return g, x, fbp, batch["meta"][0]


def main() -> None:
    args = parse_args()
    configure_cuda()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    g, x_true, fbp, meta = load_study(args.data_root, args.quality, args.mode, args.study_idx, device, load_fbp=args.fbp_init)
    T = len(g)
    shape = tuple(x_true[0].shape[1:])

    model_config = HLPDConfig(
        unrolled=HLPDUnrolledParams(num_iters=args.num_iters, shared=args.shared_iters, grad_checkpointing=args.grad_checkpointing),
        block=HLPDBlockParams(recurrent=args.recurrent, time_steps=T, shared=args.shared_time),
        conv_stack=ConvStackParams(hidden_channels=args.hidden_channels),
        deform=DeformParams(N=args.integration_steps, integration="euler"),
        normalization=NormalizationParams(operator=args.normalize_operator, intensity=args.intensity),
        zero_init=not args.fbp_init,
        loss=LossParams(domain=args.domain, lambda_coupling=args.lambda_coupling, lambda_reg=args.lambda_reg, helmholtz=tuple(args.helmholtz)),
    )
    ray_trafo_params = simulation_ray_trafo_params(args.data_root)

    config = {
        "study_id": meta["study_id"],
        **model_config.to_dict(),
        "ray_trafo": asdict(ray_trafo_params),
        "optimizer": {"name": "Adam", "lr": args.lr, "steps": args.steps},
        "seed": args.seed,
    }
    run_args = {"device": str(device), "profile": args.profile, "out_dir": str(args.out_dir)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with start_run("hlpd_overfit", config, log_dir=args.out_dir, args=run_args) as run:

        model, loss_fn = build_hlpd(model_config, ray_trafo_params, T=T, shape=shape, device=device, mu_water=simulation_mu_water(args.data_root))  # type: ignore[arg-type]
        set_geometry(model, loss_fn, meta)
        if args.normalize_operator:
            logger.info("Operator norms ||T_t||: %s", ", ".join(f"{n:.4g}" for n in model.operator_norms))
        y = g if args.domain == "projection" else x_true

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info("Study %s, shape %s, %d time bins, %d trainable parameters.", meta["study_id"], shape, T, sum(p.numel() for p in model.parameters()))

        step_stats = StepStats(device)
        prof = make_profiler(args.out_dir / "profile", *args.profile_steps) if args.profile else None
        if prof is not None:
            prof.start()

        plot_dir = args.plot_dir / run.run_id
        zero_prediction_mse = float(torch.stack([xt.square().mean() for xt in x_true]).mean())  # baseline for image_mse
        device_total_gb = torch.cuda.mem_get_info(device)[1] / 1e9 if device.type == "cuda" else None
        timeline = StepTimeline(device)
        history: list[dict[str, float]] = []
        if args.plot:
            logger.info("Writing live plots to %s", plot_dir)

        for step in range(args.steps):
            # NOTE: step 0 is skipped, as it includes one-off costs (cuDNN autotuning, growth of the allocator cache).
            record = args.plot and (step == 1 or (step > 0 and step % args.plot_every == 0))
            step_stats.start()
            with timeline.record() if record else nullcontext():
                optimizer.zero_grad(set_to_none=True)

                with region("step/forward"):
                    w, f, v = model(g, fbp)
                with region("step/loss"):
                    loss = loss_fn(w, f, v, y)
                with region("step/backward"):
                    loss.total_loss.backward()
                with region("step/optimizer"):
                    optimizer.step()

            stats = step_stats.stop()
            if record:
                stats.update(timeline.peaks)  # the timeline resets the peak memory stats at every sample, so StepStats' peaks are wrong on recorded steps
            if prof is not None:
                prof.step()

            with torch.no_grad():
                image_mse = torch.stack([torch.mean((ft - xt) ** 2) for ft, xt in zip(f, x_true)]).mean().item()
                warp_mse = torch.stack([torch.mean((wt - xt) ** 2) for wt, xt in zip(w, x_true)]).mean().item()
            metrics = {
                "total_loss": loss.total_loss.item(),
                "data_loss": loss.data_loss.item(),
                "coupling_loss": loss.coupling_loss.item(),
                "regularization_loss": loss.regularization_loss.item(),
                "image_mse": image_mse,
                "warp_mse": warp_mse,
                **stats,
            }
            run.log_metrics(step, **metrics)
            history.append({"step": step, **metrics})
            last = step == args.steps - 1
            if step % args.log_every == 0 or last:
                logger.info("step %d: %s", step, ", ".join(f"{k}={val:.4g}" for k, val in metrics.items()))

            if args.plot:
                if record:
                    plot_gpu_memory(timeline.samples, timeline.events, plot_dir / "gpu_memory.png", step, device_total_gb)
                    plot_step_times(timeline.history, plot_dir / "step_times.png")
                if step % args.plot_every == 0 or last:
                    plot_losses(history, plot_dir / "loss.png", zero_prediction_mse)
                if step % args.gif_every == 0 or last:
                    plot_prediction_gif(f, x_true, v, model.extent, plot_dir / "prediction.gif", step, x_warped=w)

        if prof is not None:
            prof.stop()
            table = module_time_table(prof)
            (args.out_dir / "profile").mkdir(parents=True, exist_ok=True)
            (args.out_dir / "profile" / f"{run.run_id}_modules.txt").write_text(table + "\n")
            logger.info("Time per module over %d profiled steps:\n%s", args.profile_steps[2], table)

        run.results = {"final_metrics": metrics, "zero_prediction_image_mse": zero_prediction_mse}

        with torch.no_grad():
            torch.save({"f": [ft.cpu() for ft in f], "w": [wt.cpu() for wt in w], "v": [vt.cpu() for vt in v]}, args.out_dir / f"{run.run_id}_output.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
