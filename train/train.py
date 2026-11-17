"""
Trains the HLPD model with its joint reconstruction and registration loss (model/hlpd_loss.py) on the simulated studies, on one or several GPUs (DistributedDataParallel).

Each process (rank) trains on its own GPU with batch_size studies per step (training config, 1 by default), and the gradients are
averaged over the ranks, so an optimizer step sees world_size * batch_size studies. Every rank sets the geometry of its own batch before
the forward pass (studies can differ in their physical extent), which is possible since the ray transform, deformation and Helmholtz
operators are not parameters and DDP only synchronizes the gradients. The studies of a batch share one geometry, so they must have the
same extent and time steps (all studies that pass the study filter of the data pipeline do). The memory grows about linearly with the
batch size (about 19 GB per study with the default model), and the last batch of an epoch has fewer studies if batch_size does not divide
the studies of a rank. The validation and test studies are evaluated one at a time.

Run from the repo root, on all 4 GPUs (or with train/scripts/train.sh):
    torchrun --standalone --nproc_per_node=4 -m train.train --config train/configs/hlpd.json
or on a single GPU, e.g. for debugging:
    python -m train.train --config train/configs/hlpd.json --max-steps 5

The data config can restrict each split to a seeded random subset of its finished studies ("max_studies", e.g. {"train": 20, "val": 5,
"test": 5}). The chosen study ids are recorded in the run config (data.study_ids) and reused when resuming.
The validation and test studies are evaluated every val_every epochs. Only the validation loss selects the best checkpoint, the test
metrics are only logged.

Rank 0 writes
    <log-dir>/runs.jsonl, logs/<run_id>.log, metrics/<run_id>.jsonl      the run record, log and per-step metrics (see tracking/run.py)
    <checkpoint-dir>/<run_id>/epoch_<e>.pt, last.pt, best.pt             checkpoints, see save_checkpoint
    <plot-dir>/<run_id>/                                                 the live plots of tracking/training_plots.py
and each rank writes the prediction and sinogram GIFs of the training (the first --train-gif-studies), validation and test studies it
evaluates (see evaluate) to
    <plot-dir>/<run_id>/<train|val|test>/<study_id>.gif                  the latest prediction, and with --keep-gifs also
    <plot-dir>/<run_id>/<train|val|test>/<study_id>/prediction_epoch<e>_step<s>.gif
    <plot-dir>/<run_id>/<train|val|test>/<study_id>_sinogram.gif         the latest data, reprojection and their difference, and with --keep-gifs also
    <plot-dir>/<run_id>/<train|val|test>/<study_id>/sinogram_epoch<e>_step<s>.gif
The per-step metrics are averaged over the ranks, while the GPU memory, timing and step time are those of rank 0.
Resume a run with --resume <checkpoint>, which starts a new run (with its own run_id) from the checkpoint's state.

The training can run in stages (see train/stages.py), e.g. first the reconstruction nets with the data term, then the motion nets with the
coupling and regularization terms, and then all nets with all terms. Each stage freezes the nets it does not train and starts with a new
DDP wrapper (DDP only synchronizes the parameters that need gradients when it is built), optimizer and scheduler. Its best checkpoint by
validation loss is best_<stage>.pt (the losses of different stages are not comparable), and best.pt that of the last stage. A resumed
run continues in the stage of its next epoch, with the optimizer state of the checkpoint only if that is the stage it was saved in.
--start-stage <stage> instead starts at the first epoch of a stage, with only the model weights of the --resume checkpoint, e.g. to try
several motion stages from the same reconstruction stage. The motion nets of the config may then differ from the checkpoint's (the
"motion" section of the model config, see model/motion.py, e.g. U-net or LDDMM motion nets after a reconstruction stage with the default
ConvStacks): only its dual and primal nets are loaded, and the motion nets start from zero (see load_model_weights).
Without stages in the training config, the training is a single stage.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import socket
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data.config import DEFAULT_SIMULATED_ROOT
from data.loaders import RegAndReconDataset, collate_time_list
from model.gpus import configure_cuda, free_gpus, select_gpu
from train.stages import Stage, merge_loss_config, parse_stages, stage_by_name, stage_index
from tracking.profiling import StepStats
from tracking.run import start_run
from tracking.timeline import StepTimeline, region
from tracking.evaluation_plots import plot_sinogram_gif
from tracking.training_plots import plot_gpu_memory, plot_losses, plot_prediction_gif, plot_step_times

# NOTE: the model is imported in main, after the GPU of this rank is selected, as importing odl creates a CUDA context on the current GPU
# (see model/gpus.py). Imported here, every rank would put one on GPU 0, which would then no longer look free to the GPU selection.
if TYPE_CHECKING:
    from model.hlpd import HLPDModel
    from model.hlpd_loss import HLPDLoss, HLPDLossTerms

logger = logging.getLogger(__name__)

METRICS = ("total_loss", "data_loss", "coupling_loss", "registered_loss", "window_loss", "regularization_loss", "temporal_loss", "supervised_motion_loss", "image_mse", "warp_mse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("train/configs/hlpd.json"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_SIMULATED_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/media/trulssv/LDDMM/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("train/runs"))
    parser.add_argument("--plot-dir", type=Path, default=Path("test/plots"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--start-stage", default=None, help="Start at the first epoch of this training stage (see train/stages.py), with "
                        "only the model weights of the --resume checkpoint (if any), rather than where the checkpoint left off.")
    parser.add_argument("--gpus", default="auto", help="'auto' for the currently free GPUs (less than 1 GB in use and idle, see model.gpus.free_gpus), "
                        "or comma separated torch indices, e.g. '1,3'. Rank i takes the i-th of them.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per rank.")
    parser.add_argument("--log-every", type=int, default=10, help="Log the metrics every this many steps.")
    parser.add_argument("--plot-every", type=int, default=25, help="Record the GPU memory and time of the key steps of a step on rank 0 and rewrite the memory, timing and loss plots every this many steps.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Write the live plots and the prediction GIFs of the validation and test studies.")
    parser.add_argument("--gif-every", type=int, default=1, help="Write the prediction GIFs at every this many evaluations.")
    parser.add_argument("--train-gif-studies", type=int, default=None, help="Write the GIFs of the first this many training studies (in eval mode) "
                        "along with the validation and test GIFs, by default one per rank. 0 for none.")
    parser.add_argument("--keep-gifs", action=argparse.BooleanOptionalAction, default=True, help="Keep the prediction GIFs of all epochs (about 3 MB each) in <plot-dir>/<run_id>/<val|test>/<study_id>/, next to the latest one in <study_id>.gif.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps, e.g. for a smoke test.")
    return parser.parse_args()


def setup_distributed(gpus: str) -> tuple[int, int, int, torch.device]:
    """(rank, world_size, local_rank, device). Selects this rank's GPU (see model.gpus.select_gpu: by default the local_rank-th of the
    currently free GPUs) and initializes the process group when launched with torchrun, otherwise runs as a single process."""
    distributed = "RANK" in os.environ
    rank, world_size, local_rank = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])) if distributed else (0, 1, 0)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if distributed:
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        if gpus == "auto":
            # NOTE: all ranks look for the free GPUs before any of them selects one, as a selected GPU gets the rank's CUDA context (and
            # ASTRA's), which can make it look busy to a rank that looks later.
            free = free_gpus()
            dist.barrier()
            if len(free) < local_world_size:
                raise RuntimeError(f"Only {len(free)} free GPUs ({free}) for {local_world_size} processes. Choose the GPUs explicitly with --gpus.")
            gpus = ",".join(map(str, free))
    device = select_gpu(local_rank, local_world_size, gpus)
    if distributed:
        # NOTE: the ranks choose their GPUs independently, so check that no two of them share one (e.g. if a GPU got busy in between).
        chosen: list = [None] * world_size
        dist.all_gather_object(chosen, (socket.gethostname(), device.index))
        assert len(set(chosen)) == world_size, f"Several ranks chose the same GPU: {chosen}. Choose the GPUs explicitly with --gpus."
    return rank, world_size, local_rank, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_studies(dataset: RegAndReconDataset, n: int | None, seed: int) -> list[str]:
    """The ids of a seeded random subset of n of the dataset's studies (all of them if n is None or not smaller), sorted."""
    ids = sorted(d.name for d in dataset.study_dirs)
    if n is None or n >= len(ids):
        return ids
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(ids[i] for i in order[:n])


def make_datasets(data_cfg: dict, data_root: Path, load_fbp: bool, rank: int, world_size: int, study_ids: dict[str, list[str]] | None = None
                  ) -> dict[str, RegAndReconDataset]:
    """The train, val and test datasets, of the given study ids (e.g. those of a resumed run) or else of the subsets chosen by
    data_cfg["max_studies"]. Rank 0 chooses the subsets and broadcasts them, so that all ranks use the same studies even if the data
    pipeline finishes more studies in the meantime."""
    qualities = {"train": data_cfg["train_qualities"], "val": data_cfg["val_qualities"], "test": data_cfg.get("test_qualities", data_cfg["val_qualities"])}
    if study_ids is None:
        if rank == 0:
            max_studies, seed = data_cfg.get("max_studies") or {}, data_cfg.get("subset_seed", 0)
            study_ids = {split: select_studies(RegAndReconDataset(qualities=q, mode=split, data_root=data_root), max_studies.get(split), seed)
                         for split, q in qualities.items()}
        if world_size > 1:
            obj = [study_ids]
            dist.broadcast_object_list(obj, src=0)
            study_ids = obj[0]
    return {split: RegAndReconDataset(qualities=q, mode=split, data_root=data_root, load_volume=True, load_fbp=load_fbp, study_ids=study_ids[split])
            for split, q in qualities.items()}


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
                "registered_loss": loss.registered_loss.item(), "window_loss": loss.window_loss.item(),
                "regularization_loss": loss.regularization_loss.item(), "temporal_loss": loss.temporal_loss.item(),
                "supervised_motion_loss": loss.supervised_motion_loss.item(), "image_mse": image_mse.item(), "warp_mse": warp_mse.item()}


def loss_fn_terms(params) -> str:
    """The included terms of the loss parameters (LossParams of model/hlpd_loss.py) with their weights, for the log."""
    return ", ".join(f"{name} ({getattr(params, name).weight:g})" for name in params.TERMS if getattr(params, name).include)


def to_device(batch: dict, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, dict]:
    """The sinogram, volume and (if loaded, see RegAndReconDataset) FBP of a batch on the device, and the meta data of its first study,
    whose geometry (extent and time steps) all studies of the batch must share, as the model and loss use one geometry per batch."""
    metas = batch["meta"]
    geometry = lambda m: (tuple(m["extent"]), tuple(m["time_steps"]))
    assert all(geometry(m) == geometry(metas[0]) for m in metas), \
        f"The studies {batch['study_id']} of a batch differ in their extent or time steps: {[geometry(m) for m in metas]}. Use batch_size 1."
    g = [gt.to(device, non_blocking=True) for gt in batch["sinogram"]]       # T x (B, nAngles, nCol, nRow)
    x = [xt.to(device, non_blocking=True) for xt in batch["volume"]]         # T x (B, D, H, W)
    fbp = [ft.to(device, non_blocking=True) for ft in batch["fbp"]] if "fbp" in batch else None   # T x (B, D, H, W)
    return g, x, fbp, metas[0]


def plot_study_sinogram_gif(model: HLPDModel, g: list[torch.Tensor], f: list[torch.Tensor], path: Path, title: str, history_path: Path | None = None) -> None:
    """The sinogram GIF (see tracking/evaluation_plots.plot_sinogram_gif) of the data g and the reprojection of the reconstructions f
    (first batch element) at path, and with history_path also kept there."""
    ray_trafo = model.ray_trafo
    g_np = [gt[0].detach().float().cpu().numpy() for gt in g]
    reprojection = [ray_trafo.forward_bin(ft, t)[0].detach().float().cpu().numpy() for t, ft in enumerate(f)]
    angles = [(float(d.min_pt[0]), float(d.max_pt[0])) for d in ray_trafo.angular_discretizations]
    plot_sinogram_gif(g_np, angles, (ray_trafo.DetectorColExtent, ray_trafo.DetectorRowExtent), history_path or path, title, reprojection=reprojection)
    if history_path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history_path, path)


@torch.no_grad()
def evaluate(model: HLPDModel, loss_fn: HLPDLoss, loader: DataLoader, domain: str, device: torch.device, world_size: int,
             gif_dir: Path | None = None, epoch: int = 0, step: int = 0, keep_gifs: bool = True) -> tuple[dict[str, float], int]:
    """Mean metrics over the studies of the loader (each rank evaluates its own shard, without DDP) and the number of studies.
    With gif_dir, the prediction GIF of each of this rank's studies is written to gif_dir/<study_id>.gif and its sinogram GIF (the data,
    the reprojection of the reconstruction and their difference) to gif_dir/<study_id>_sinogram.gif, and with keep_gifs both are also
    kept in gif_dir/<study_id>/."""
    from model.build import set_geometry

    model.eval()
    sums = torch.zeros(len(METRICS) + 2, device=device, dtype=torch.float64)  # metrics, zero prediction mse, count
    for batch in loader:
        g, x_true, fbp, meta = to_device(batch, device)
        set_geometry(model, loss_fn, meta)
        w, f, v = model(g, fbp)
        loss = loss_fn(w, f, v, g if domain == "projection" else x_true)
        metrics = step_metrics(loss, w, f, x_true)
        zero_mse = torch.stack([xt.square().mean() for xt in x_true]).mean()
        sums += torch.tensor([*metrics.values(), zero_mse.item(), 1.0], device=device, dtype=torch.float64)
        if gif_dir is not None:
            study_id = batch["study_id"][0]
            plot_prediction_gif(f, x_true, v, model.extent, gif_dir / f"{study_id}.gif", step, x_warped=w, epoch=epoch,
                                history_dir=gif_dir / study_id if keep_gifs else None)
            plot_study_sinogram_gif(model, g, f, gif_dir / f"{study_id}_sinogram.gif", f"{study_id}, epoch {epoch}, step {step}",
                                    history_path=gif_dir / study_id / f"sinogram_epoch{epoch}_step{step}.gif" if keep_gifs else None)
    if world_size > 1:
        dist.all_reduce(sums)
    model.train()
    n = int(sums[-1].item())
    means = (sums[:-1] / max(n, 1)).tolist()
    return {**dict(zip(METRICS, means[:-1])), "zero_prediction_image_mse": means[-1]}, n


def save_checkpoint(path: Path, model: HLPDModel, optimizer: torch.optim.Optimizer, scheduler, epoch: int, step: int, best_val: float, config: dict, run_id: str,
                    stage: int = 0, stage_name: str = "train") -> None:
    """best_val is the best validation loss of the stage (the index of the training stage of the epoch, see train/stages.py), and the
    optimizer and scheduler states are those of the stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "stage": stage,
        "stage_name": stage_name,
        "config": config,
        "run_id": run_id,
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
    }, tmp)
    os.replace(tmp, path)


def load_model_weights(model: HLPDModel, checkpoint: dict, motion, new_motion_ok: bool) -> None:
    """Loads the checkpoint's model weights. If the checkpoint was trained with another motion config (MotionParams of model/motion.py,
    e.g. a reconstruction stage with the default ConvStacks, resumed with U-net or LDDMM motion nets), only its dual and primal nets are
    loaded and the motion nets keep their zero initialization. That is only allowed with --start-stage (new_motion_ok), as a resumed
    stage would continue an optimizer state of the old motion nets."""
    from model.motion import MotionParams

    saved = MotionParams.from_dict(checkpoint["config"].get("motion") or {})
    if saved == motion:
        model.load_state_dict(checkpoint["model"])
        return
    assert new_motion_ok, (f"The checkpoint's motion config {asdict(saved)} differs from the config's {asdict(motion)}. Start a stage with "
                           "--start-stage to load only its dual and primal nets, with new motion nets.")
    is_motion = lambda key: ".motion_net." in key
    state = {k: v for k, v in checkpoint["model"].items() if not is_motion(k)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(is_motion(k) for k in missing), f"Unexpected keys {unexpected} or missing keys {[k for k in missing if not is_motion(k)]}."
    trained = any(v.any() for k, v in checkpoint["model"].items() if is_motion(k) and v.is_floating_point() and k.endswith("weight"))
    logger.warning("The checkpoint's motion config %s differs from the config's %s: loaded only its dual and primal nets, the motion nets are "
                   "new and zero initialized%s.", asdict(saved), asdict(motion), " (its motion nets were trained, and are discarded)" if trained else "")


def main() -> None:
    args = parse_args()
    configure_cuda()
    rank, world_size, local_rank, device = setup_distributed(args.gpus)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format=f"%(asctime)s [rank {rank}] %(levelname)s %(message)s")
    logger.info("Rank %d of %d on GPU %d (--gpus %s).", rank, world_size, device.index, args.gpus)
    from model.build import HLPDConfig, LossParams, build_hlpd, set_geometry, simulation_mu_water, simulation_ray_trafo_params

    with open(args.config) as f:
        file_config = json.load(f)
    model_config = HLPDConfig.from_dict(file_config["model"])
    data_cfg, opt_cfg, sched_cfg, train_cfg = file_config["data"], file_config["optimizer"], file_config.get("scheduler"), file_config["training"]
    stages = parse_stages(train_cfg)
    total_epochs = stages[-1].end
    # NOTE: the loss of each stage is built from the loss of the model config (as a dict of its sections) and the changes of the stage, so
    # that they are checked before the data is loaded.
    base_loss = asdict(model_config.loss)
    stage_losses = [LossParams.from_dict(merge_loss_config(base_loss, stage.loss)) for stage in stages]
    seed_everything(train_cfg["seed"])  # NOTE: the same seed on all ranks, DDP also broadcasts rank 0's initial parameters

    checkpoint = torch.load(args.resume, map_location=device, weights_only=False) if args.resume is not None else None

    # Data: each rank trains on its own part of the shuffled studies (DistributedSampler) and evaluates every world_size-th validation
    # and test study. A resumed run uses the studies of the checkpoint's run.
    # NOTE: the FBPs stored by the data pipeline are only needed (and loaded) for the FBP initialization.
    load_fbp = not model_config.zero_init
    resumed_ids = checkpoint["config"]["data"].get("study_ids") if checkpoint is not None else None
    datasets = make_datasets(data_cfg, args.data_root, load_fbp, rank, world_size, study_ids=resumed_ids)
    train_set, val_set, test_set = datasets["train"], datasets["val"], datasets["test"]
    assert len(train_set) > 0, f"No finished training studies of qualities {data_cfg['train_qualities']} under {args.data_root}."
    T, shape = study_shape(train_set)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=train_cfg["seed"])
    batch_size = train_cfg.get("batch_size", 1)
    assert batch_size >= 1, f"batch_size must be at least 1, but got {batch_size}."
    # NOTE: only the training batches have batch_size studies, evaluate and its GIFs take one study at a time.
    loader_kwargs = {"batch_size": 1, "collate_fn": collate_time_list, "num_workers": args.num_workers, "pin_memory": True, "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_set, sampler=train_sampler, **{**loader_kwargs, "batch_size": batch_size})
    val_loader = DataLoader(Subset(val_set, range(rank, len(val_set), world_size)), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(test_set, range(rank, len(test_set), world_size)), shuffle=False, **loader_kwargs)
    n_train_gifs = min(world_size if args.train_gif_studies is None else args.train_gif_studies, len(train_set))
    train_gif_loader = DataLoader(Subset(train_set, range(rank, n_train_gifs, world_size)), shuffle=False, **loader_kwargs)

    ray_trafo_params = simulation_ray_trafo_params(args.data_root)
    model, loss_fn = build_hlpd(model_config, ray_trafo_params, T=T, shape=shape, device=device, mu_water=simulation_mu_water(args.data_root))

    def start_stage(stage: Stage, loss_params: LossParams) -> tuple[torch.nn.Module, torch.optim.Optimizer, object]:
        """Freezes the nets the stage does not train, sets its loss terms, and returns a new DDP wrapper, optimizer and scheduler."""
        model.set_trainable(stage.train)
        loss_fn.set_params(loss_params)
        params = [p for p in model.parameters() if p.requires_grad]
        ddp = DDP(model, device_ids=[device.index]) if world_size > 1 else model
        optimizer = getattr(torch.optim, opt_cfg["name"])(params, **{**opt_cfg.get("params", {}), **({"lr": stage.lr} if stage.lr is not None else {})})
        scheduler = getattr(torch.optim.lr_scheduler, sched_cfg["name"])(optimizer, **sched_cfg.get("params", {})) if sched_cfg else None  # stepped once per epoch
        return ddp, optimizer, scheduler

    start_epoch, step, best_val = 0, 0, math.inf
    if checkpoint is not None:
        load_model_weights(model, checkpoint, model_config.motion, new_motion_ok=args.start_stage is not None)
        start_epoch, step, best_val = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["best_val"]
    if args.start_stage is not None:
        start_epoch, best_val = stages[stage_by_name(stages, args.start_stage)].start, math.inf
    # NOTE: the optimizer and scheduler states of the checkpoint are restored when the training starts in the stage they were saved in.
    resumed_state = checkpoint if checkpoint is not None and args.start_stage is None else None

    config = {
        **model_config.to_dict(),
        "ray_trafo": asdict(ray_trafo_params),
        "data": {**data_cfg, "data_root": str(args.data_root), "train_studies": len(train_set), "val_studies": len(val_set), "test_studies": len(test_set),
                 "study_ids": {split: [d.name for d in dataset.study_dirs] for split, dataset in datasets.items()}},
        "optimizer": opt_cfg,
        "scheduler": sched_cfg,
        "training": {**train_cfg, "epochs": total_epochs, "stages": [asdict(s) for s in stages], "batch_size": batch_size, "world_size": world_size, "studies_per_step": world_size * batch_size},
    }
    run_args = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "world_size": world_size, "gpu_rank0": device.index}

    args.log_dir.mkdir(parents=True, exist_ok=True)
    with start_run("hlpd_train", config, log_dir=args.log_dir, args=run_args) if rank == 0 else nullcontext() as run:
        run_id = run.run_id if run is not None else ""
        if world_size > 1:  # the other ranks write GIFs into the run's plot directory too
            obj = [run_id]
            dist.broadcast_object_list(obj, src=0)
            run_id = obj[0]
        checkpoint_dir, plot_dir = args.checkpoint_dir / run_id, args.plot_dir / run_id
        if rank == 0:
            logger.info("%d GPUs, %d training, %d validation and %d test studies of shape %s with %d time bins, %d trainable parameters.",
                        world_size, len(train_set), len(val_set), len(test_set), shape, T, sum(p.numel() for p in model.parameters()))
            logger.info("Checkpoints in %s, plots in %s.", checkpoint_dir, plot_dir)
            if args.resume is not None:
                logger.info("Resumed from %s at epoch %d, step %d%s.", args.resume, start_epoch, step,
                            f", only its weights, at the start of stage '{args.start_stage}'" if args.start_stage is not None else "")
            if len(val_set) == 0:
                logger.warning("No validation studies, the best checkpoint and the prediction GIF use the training studies instead.")

        step_stats = StepStats(device)
        timeline = StepTimeline(device)
        device_total_gb = torch.cuda.mem_get_info(device)[1] / 1e9
        history: list[dict[str, float]] = []
        val_history: list[dict[str, float]] = []
        zero_prediction_mse = None
        val = test = None
        n_evals = 0
        done = False
        current = None  # the index of the current stage
        ddp_model: torch.nn.Module | None = None

        for epoch in range(start_epoch, total_epochs):
            if stage_index(stages, epoch) != current:
                current = stage_index(stages, epoch)
                stage = stages[current]
                # NOTE: the previous stage's DDP wrapper is collected before the next one is built, as its reducer hooks the gradients
                # of the parameters it synchronizes, which the next stage may train as well.
                ddp_model = optimizer = scheduler = None
                gc.collect()
                ddp_model, optimizer, scheduler = start_stage(stage, stage_losses[current])
                if resumed_state is not None and resumed_state.get("stage", 0) == current:
                    saved = resumed_state["optimizer"]["param_groups"]
                    if [len(g["params"]) for g in saved] == [len(g["params"]) for g in optimizer.param_groups]:
                        optimizer.load_state_dict(resumed_state["optimizer"])
                        if scheduler is not None and resumed_state["scheduler"] is not None:
                            scheduler.load_state_dict(resumed_state["scheduler"])
                    else:
                        logger.warning("The optimizer state of %s is not that of the nets of stage '%s', so it starts anew.", args.resume, stage.name)
                else:
                    best_val = math.inf  # the losses of different stages are not comparable
                resumed_state = None
                if rank == 0:
                    logger.info("Stage %d of %d '%s' (epochs %d to %d): training the %s nets (%d parameters, lr %g) with the loss terms %s%s.",
                                current + 1, len(stages), stage.name, stage.start, stage.end - 1, ", ".join(stage.train),
                                sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer.param_groups[0]["lr"],
                                loss_fn_terms(stage_losses[current]),
                                ", skipping the motion nets and warps (frozen and zero, so v = 0 and w_{t+1} = f_t)" if model.skip_motion else "")
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
                    run.log_metrics(step, epoch=epoch, stage=current, lr=lr, **metrics, **stats)
                    history.append({"step": step, "stage": current, **metrics})
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
            last_epoch = done or epoch == total_epochs - 1

            # Validation and test, with the prediction GIFs of their studies (or of rank 0's last training study if there are no validation studies).
            if (epoch + 1) % train_cfg["val_every"] == 0 or last_epoch:
                gifs = args.plot and (n_evals % args.gif_every == 0 or last_epoch)
                n_evals += 1
                eval_kwargs = {"epoch": epoch, "step": step, "keep_gifs": args.keep_gifs}
                val, n_val = evaluate(model, loss_fn, val_loader, model_config.loss.domain, device, world_size, gif_dir=plot_dir / "val" if gifs else None,
                                      **eval_kwargs) if len(val_set) > 0 else (None, 0)
                test, n_test = evaluate(model, loss_fn, test_loader, model_config.loss.domain, device, world_size, gif_dir=plot_dir / "test" if gifs else None,
                                        **eval_kwargs) if len(test_set) > 0 else (None, 0)
                if gifs and n_train_gifs > 0:  # only for the GIFs, the training metrics are those of the steps
                    evaluate(model, loss_fn, train_gif_loader, model_config.loss.domain, device, world_size, gif_dir=plot_dir / "train", **eval_kwargs)
                if rank == 0:
                    epoch_means = {k: s / max(n_steps, 1) for k, s in epoch_sums.items()}
                    logger.info("epoch %d: train %s", epoch, ", ".join(f"{k}={v:.4g}" for k, v in epoch_means.items()))
                    if val is not None:
                        logger.info("epoch %d: validation (%d studies) %s", epoch, n_val, ", ".join(f"{k}={v:.4g}" for k, v in val.items()))
                        run.log_metrics(step, epoch=epoch, stage=current, **{f"val_{k}": v for k, v in val.items()})
                        val_history.append({"step": step, "stage": current, **val})
                        zero_prediction_mse = val["zero_prediction_image_mse"]
                    if test is not None:
                        logger.info("epoch %d: test (%d studies) %s", epoch, n_test, ", ".join(f"{k}={v:.4g}" for k, v in test.items()))
                        run.log_metrics(step, epoch=epoch, stage=current, **{f"test_{k}": v for k, v in test.items()})
                    if args.plot:
                        plot_losses(history, plot_dir / "loss.png", zero_prediction_mse, val_history)
                        if gifs and val is None:
                            plot_prediction_gif(f, x_true, v, model.extent, plot_dir / "prediction.gif", step, x_warped=w, epoch=epoch,
                                                history_dir=plot_dir / "gifs" if args.keep_gifs else None)

            # Checkpoints: every checkpoint_every epochs, the last one, and the best of the stage by validation (or training) loss, as
            # best_<stage>.pt with several stages, and as best.pt in the last stage.
            if rank == 0:
                monitored = val["total_loss"] if val is not None else epoch_sums["total_loss"] / max(n_steps, 1)
                ckpt_args = (model, optimizer, scheduler, epoch, step)
                ckpt_kwargs = {"stage": current, "stage_name": stage.name}
                if monitored < best_val:
                    best_val = monitored
                    if len(stages) > 1:
                        save_checkpoint(checkpoint_dir / f"best_{stage.name}.pt", *ckpt_args, best_val, config, run_id, **ckpt_kwargs)
                    if current == len(stages) - 1:
                        save_checkpoint(checkpoint_dir / "best.pt", *ckpt_args, best_val, config, run_id, **ckpt_kwargs)
                if (epoch + 1) % train_cfg["checkpoint_every"] == 0 or last_epoch or epoch == stage.end - 1:
                    save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", *ckpt_args, best_val, config, run_id, **ckpt_kwargs)
                save_checkpoint(checkpoint_dir / "last.pt", *ckpt_args, best_val, config, run_id, **ckpt_kwargs)
            if done:
                break

        if rank == 0:
            run.results = {"epochs": epoch + 1, "steps": step, "best_val_total_loss": best_val, "stage": stages[current].name if current is not None else None, "final_val": val, "final_test": test, "checkpoint_dir": str(checkpoint_dir)}

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
