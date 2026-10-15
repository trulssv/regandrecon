"""
Trains the HLPD model with its joint reconstruction and registration loss (model/hlpd_loss.py) on the simulated studies, on one or several GPUs (DistributedDataParallel).

Each process (rank) trains on its own GPU with one study per step, and the gradients are averaged over the ranks, so an optimizer
step sees world_size studies. Every rank sets the geometry of its own study before the forward pass (the studies differ in their
physical extent), which is possible since the ray transform, deformation and Helmholtz operators are not parameters and DDP only
synchronizes the gradients.

Run from the repo root, on all 4 GPUs:
    torchrun --standalone --nproc_per_node=4 -m train.train --config train/configs/hlpd.json
or on a single GPU, e.g. for debugging:
    python -m train.train --config train/configs/hlpd.json --max-steps 5

Rank 0 writes
    <log-dir>/runs.jsonl, logs/<run_id>.log, metrics/<run_id>.jsonl      the run record, log and per-step metrics (see tracking/run.py)
    <checkpoint-dir>/<run_id>/epoch_<e>.pt, last.pt, best.pt             checkpoints, see save_checkpoint
    <plot-dir>/<run_id>/                                                 the live plots of tracking/training_plots.py
The per-step metrics are averaged over the ranks, while the GPU memory, timing and step time are those of rank 0.
Resume a run with --resume <checkpoint>, which starts a new run (with its own run_id) from the checkpoint's state.
"""
import argparse
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RegAndReconDataset, collate_time_list
from model.build import HLPDConfig, build_hlpd, configure_cuda, set_geometry, simulation_mu_water, simulation_ray_trafo_params
from model.hlpd import HLPDModel
from model.hlpd_loss import HLPDLoss, HLPDLossTerms
from tracking.profiling import StepStats
from tracking.run import start_run
from tracking.timeline import StepTimeline, region
from tracking.training_plots import plot_gpu_memory, plot_losses, plot_prediction_gif, plot_step_times

logger = logging.getLogger(__name__)

METRICS = ("total_loss", "data_loss", "coupling_loss", "regularization_loss", "image_mse", "warp_mse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("train/configs/hlpd.json"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_SIMULATED_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/media/trulssv/LDDMM/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("train/runs"))
    parser.add_argument("--plot-dir", type=Path, default=Path("test/plots"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per rank.")
    parser.add_argument("--log-every", type=int, default=10, help="Log the metrics every this many steps.")
    parser.add_argument("--plot-every", type=int, default=25, help="Record the GPU memory and time of the key steps of a step on rank 0 and rewrite the memory, timing and loss plots every this many steps.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Write the live plots. The prediction GIF is written at every validation.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps, e.g. for a smoke test.")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    """(rank, world_size, local_rank). Initializes the process group when launched with torchrun, otherwise runs as a single process."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return rank, world_size, local_rank


def select_gpu(local_rank: int) -> torch.device:
    """The GPU of this rank, for torch and for ASTRA.
    NOTE: odl only selects the ASTRA GPU when the reconstruction space lives on a GPU (see odl/applications/tomo/backends/astra_cuda.py),
    which ours does not, so without this every rank would run its ray transforms on GPU 0."""
    import astra

    torch.cuda.set_device(local_rank)
    astra.set_gpu_index(local_rank)
    return torch.device("cuda", local_rank)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def study_shape(dataset: RegAndReconDataset) -> tuple[int, tuple[int, int, int]]:
    """The number of time bins and the volume shape, which must be the same for all studies (the model is built for one shape)."""
    shapes = set()
    for study_dir in dataset.study_dirs:
        with open(study_dir / "meta.json") as f:
            meta = json.load(f)
        shapes.add((len(meta["time_steps"]), tuple(meta["shape"])))
    assert len(shapes) == 1, f"All studies must have the same number of time bins and shape, but found {shapes}."
    return shapes.pop()


def all_reduce_mean(values: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    if world_size == 1:
        return values
    tensor = torch.tensor(list(values.values()), device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    return dict(zip(values, (tensor / world_size).tolist()))


def step_metrics(loss: HLPDLossTerms, w: list[torch.Tensor], f: list[torch.Tensor], x_true: list[torch.Tensor]) -> dict[str, float]:
    """The loss terms, and the image domain errors of the reconstructions f (image_mse) and of their predictions w by the motion (warp_mse)."""
    with torch.no_grad():
        image_mse = torch.stack([torch.mean((ft - xt) ** 2) for ft, xt in zip(f, x_true)]).mean()
        warp_mse = torch.stack([torch.mean((wt - xt) ** 2) for wt, xt in zip(w, x_true)]).mean()
        return {"total_loss": loss.total_loss.item(), "data_loss": loss.data_loss.item(), "coupling_loss": loss.coupling_loss.item(),
                "regularization_loss": loss.regularization_loss.item(), "image_mse": image_mse.item(), "warp_mse": warp_mse.item()}


def to_device(batch: dict, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, dict]:
    """The sinogram, volume and (if loaded, see RegAndReconDataset) FBP of a batch on the device, and the meta data of its first study."""
    g = [gt.to(device, non_blocking=True) for gt in batch["sinogram"]]       # T x (1, nAngles, nCol, nRow)
    x = [xt.to(device, non_blocking=True) for xt in batch["volume"]]         # T x (1, D, H, W)
    fbp = [ft.to(device, non_blocking=True) for ft in batch["fbp"]] if "fbp" in batch else None   # T x (1, D, H, W)
    return g, x, fbp, batch["meta"][0]


@torch.no_grad()
def validate(model: HLPDModel, loss_fn: HLPDLoss, loader: DataLoader, domain: str, device: torch.device, world_size: int) -> tuple[dict[str, float], int, dict | None]:
    """Mean metrics over the validation studies (each rank evaluates its own shard, without DDP), the number of studies, and the
    outputs of this rank's first study for the prediction GIF."""
    model.eval()
    sums = torch.zeros(len(METRICS) + 2, device=device, dtype=torch.float64)  # metrics, zero prediction mse, count
    first = None
    for batch in loader:
        g, x_true, fbp, meta = to_device(batch, device)
        set_geometry(model, loss_fn, meta)
        w, f, v = model(g, fbp)
        loss = loss_fn(w, f, v, g if domain == "projection" else x_true)
        metrics = step_metrics(loss, w, f, x_true)
        zero_mse = torch.stack([xt.square().mean() for xt in x_true]).mean()
        sums += torch.tensor([*metrics.values(), zero_mse.item(), 1.0], device=device, dtype=torch.float64)
        if first is None:
            first = {"f": [ft.cpu() for ft in f], "w": [wt.cpu() for wt in w], "x_true": [xt.cpu() for xt in x_true], "v": [vt.cpu() for vt in v], "extent": model.extent}
    if world_size > 1:
        dist.all_reduce(sums)
    model.train()
    n = int(sums[-1].item())
    means = (sums[:-1] / max(n, 1)).tolist()
    return {**dict(zip(METRICS, means[:-1])), "zero_prediction_image_mse": means[-1]}, n, first


def save_checkpoint(path: Path, model: HLPDModel, optimizer: torch.optim.Optimizer, scheduler, epoch: int, step: int, best_val: float, config: dict, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "config": config,
        "run_id": run_id,
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
    }, tmp)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    configure_cuda()
    rank, world_size, local_rank = setup_distributed()
    device = select_gpu(local_rank)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format=f"%(asctime)s [rank {rank}] %(levelname)s %(message)s")

    with open(args.config) as f:
        file_config = json.load(f)
    model_config = HLPDConfig.from_dict(file_config["model"])
    data_cfg, opt_cfg, sched_cfg, train_cfg = file_config["data"], file_config["optimizer"], file_config.get("scheduler"), file_config["training"]
    seed_everything(train_cfg["seed"])  # NOTE: the same seed on all ranks, DDP also broadcasts rank 0's initial parameters

    # Data: each rank trains on its own part of the shuffled studies (DistributedSampler) and validates on every world_size-th study.
    # NOTE: the FBPs stored by the data pipeline are only needed (and loaded) for the FBP initialization.
    load_fbp = not model_config.zero_init
    train_set = RegAndReconDataset(qualities=data_cfg["train_qualities"], mode="train", data_root=args.data_root, load_volume=True, load_fbp=load_fbp)
    val_set = RegAndReconDataset(qualities=data_cfg["val_qualities"], mode="val", data_root=args.data_root, load_volume=True, load_fbp=load_fbp)
    assert len(train_set) > 0, f"No finished training studies of qualities {data_cfg['train_qualities']} under {args.data_root}."
    T, shape = study_shape(train_set)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=train_cfg["seed"])
    loader_kwargs = {"batch_size": 1, "collate_fn": collate_time_list, "num_workers": args.num_workers, "pin_memory": True, "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(Subset(val_set, range(rank, len(val_set), world_size)), shuffle=False, **loader_kwargs)

    ray_trafo_params = simulation_ray_trafo_params(args.data_root)
    model, loss_fn = build_hlpd(model_config, ray_trafo_params, T=T, shape=shape, device=device, mu_water=simulation_mu_water(args.data_root))
    ddp_model = DDP(model, device_ids=[local_rank]) if world_size > 1 else model
    optimizer = getattr(torch.optim, opt_cfg["name"])(model.parameters(), **opt_cfg.get("params", {}))
    scheduler = getattr(torch.optim.lr_scheduler, sched_cfg["name"])(optimizer, **sched_cfg.get("params", {})) if sched_cfg else None  # stepped once per epoch

    start_epoch, step, best_val = 0, 0, math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, step, best_val = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["best_val"]

    config = {
        **model_config.to_dict(),
        "ray_trafo": asdict(ray_trafo_params),
        "data": {**data_cfg, "data_root": str(args.data_root), "train_studies": len(train_set), "val_studies": len(val_set)},
        "optimizer": opt_cfg,
        "scheduler": sched_cfg,
        "training": {**train_cfg, "world_size": world_size, "studies_per_step": world_size},
    }
    run_args = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "world_size": world_size}

    args.log_dir.mkdir(parents=True, exist_ok=True)
    with start_run("hlpd_train", config, log_dir=args.log_dir, args=run_args) if rank == 0 else nullcontext() as run:
        run_id = run.run_id if run is not None else ""
        checkpoint_dir, plot_dir = args.checkpoint_dir / run_id, args.plot_dir / run_id
        if rank == 0:
            logger.info("%d GPUs, %d training and %d validation studies of shape %s with %d time bins, %d trainable parameters.",
                        world_size, len(train_set), len(val_set), shape, T, sum(p.numel() for p in model.parameters()))
            logger.info("Checkpoints in %s, plots in %s.", checkpoint_dir, plot_dir)
            if args.resume is not None:
                logger.info("Resumed from %s at epoch %d, step %d.", args.resume, start_epoch, step)
            if len(val_set) == 0:
                logger.warning("No validation studies, the best checkpoint and the prediction GIF use the training studies instead.")

        step_stats = StepStats(device)
        timeline = StepTimeline(device)
        device_total_gb = torch.cuda.mem_get_info(device)[1] / 1e9
        history: list[dict[str, float]] = []
        val_history: list[dict[str, float]] = []
        zero_prediction_mse = None
        val = None
        done = False

        for epoch in range(start_epoch, train_cfg["epochs"]):
            train_sampler.set_epoch(epoch)
            epoch_sums = dict.fromkeys(METRICS, 0.0)
            n_steps = 0

            for batch in train_loader:
                g, x_true, fbp, meta = to_device(batch, device)
                set_geometry(model, loss_fn, meta)
                y = g if model_config.loss.domain == "projection" else x_true

                # NOTE: the first step is not recorded, as it includes one-off costs (cuDNN autotuning, growth of the allocator cache).
                record = rank == 0 and args.plot and step > 0 and (step == 1 or step % args.plot_every == 0)
                step_stats.start()
                with timeline.record() if record else nullcontext():
                    optimizer.zero_grad(set_to_none=True)
                    with region("step/forward"):
                        w, f, v = ddp_model(g, fbp)
                    with region("step/loss"):
                        loss = loss_fn(w, f, v, y)
                    with region("step/backward"):
                        loss.total_loss.backward()
                    with region("step/optimizer"):
                        if train_cfg.get("grad_clip_norm"):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
                        optimizer.step()
                stats = step_stats.stop()
                if record:
                    stats.update(timeline.peaks)  # the timeline resets the peak memory stats at every sample

                metrics = all_reduce_mean(step_metrics(loss, w, f, x_true), device, world_size)
                for key in METRICS:
                    epoch_sums[key] += metrics[key]
                n_steps += 1

                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    run.log_metrics(step, epoch=epoch, lr=lr, **metrics, **stats)
                    history.append({"step": step, **metrics})
                    if step % args.log_every == 0:
                        logger.info("epoch %d step %d: %s", epoch, step, ", ".join(f"{k}={val:.4g}" for k, val in {**metrics, **stats}.items()))
                    if args.plot:
                        if record:
                            plot_gpu_memory(timeline.samples, timeline.events, plot_dir / "gpu_memory.png", step, device_total_gb)
                            plot_step_times(timeline.history, plot_dir / "step_times.png")
                        if step % args.plot_every == 0:
                            plot_losses(history, plot_dir / "loss.png", zero_prediction_mse, val_history)

                step += 1
                if args.max_steps is not None and step >= args.max_steps:
                    done = True
                    break

            if scheduler is not None:
                scheduler.step()
            last_epoch = done or epoch == train_cfg["epochs"] - 1

            # Validation, and the prediction GIF of rank 0's first validation study (or of its last training study if there are none).
            if (epoch + 1) % train_cfg["val_every"] == 0 or last_epoch:
                val, n_val, first = validate(model, loss_fn, val_loader, model_config.loss.domain, device, world_size) if len(val_set) > 0 else (None, 0, None)
                if rank == 0:
                    epoch_means = {k: s / max(n_steps, 1) for k, s in epoch_sums.items()}
                    logger.info("epoch %d: train %s", epoch, ", ".join(f"{k}={v:.4g}" for k, v in epoch_means.items()))
                    if val is not None:
                        logger.info("epoch %d: validation (%d studies) %s", epoch, n_val, ", ".join(f"{k}={v:.4g}" for k, v in val.items()))
                        run.log_metrics(step, epoch=epoch, **{f"val_{k}": v for k, v in val.items()})
                        val_history.append({"step": step, **val})
                        zero_prediction_mse = val["zero_prediction_image_mse"]
                    if args.plot:
                        plot_losses(history, plot_dir / "loss.png", zero_prediction_mse, val_history)
                        if first is None:
                            first = {"f": f, "w": w, "x_true": x_true, "v": v, "extent": model.extent}
                        plot_prediction_gif(first["f"], first["x_true"], first["v"], first["extent"], plot_dir / "prediction.gif", step, x_warped=first["w"])

            # Checkpoints: every checkpoint_every epochs, the last one, and the best by validation (or training) loss.
            if rank == 0:
                monitored = val["total_loss"] if val is not None else epoch_sums["total_loss"] / max(n_steps, 1)
                ckpt_args = (model, optimizer, scheduler, epoch, step)
                if monitored < best_val:
                    best_val = monitored
                    save_checkpoint(checkpoint_dir / "best.pt", *ckpt_args, best_val, config, run_id)
                if (epoch + 1) % train_cfg["checkpoint_every"] == 0 or last_epoch:
                    save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", *ckpt_args, best_val, config, run_id)
                save_checkpoint(checkpoint_dir / "last.pt", *ckpt_args, best_val, config, run_id)
            if done:
                break

        if rank == 0:
            run.results = {"epochs": epoch + 1, "steps": step, "best_val_total_loss": best_val, "final_val": val, "checkpoint_dir": str(checkpoint_dir)}

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
