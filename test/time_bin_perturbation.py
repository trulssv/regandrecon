"""
Shared parts of the time bin perturbation tests of a trained HLPD model with per time bin nets (block.shared = False):
    test/eval_time_shift.py         shifts the phases cyclically (the action of Z/T on the time bins)
    test/eval_net_substitution.py   replaces the nets of all time bins by the net of a single bin, or by their weight average

Both evaluate a checkpoint of train/train.py, by default the last checkpoint of the latest run under the checkpoint directory, on the
studies of one of its splits, as train/evaluate.py does. The metrics are per time bin, so that phases of different difficulty are not
averaged away, and their means are also reported.

NOTE: the model is only imported once the GPU is selected (see train/train.py), so model.* is imported inside the functions here.
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from model.hlpd import HLPDBlock, HLPDModel
    from model.hlpd_loss import HLPDLoss

logger = logging.getLogger(__name__)

NET_GROUPS = ("dual", "primal", "motion")       # as HLPDModel.NET_GROUPS
DEFAULT_CHECKPOINT_DIR = Path("/media/trulssv/LDDMM/checkpoints")
DEFAULT_PLOT_DIR = Path("test/plots")


# ----------------------------------------------------------------- arguments and loading

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", nargs="?", default=None,
                        help="Run id (under --checkpoint-dir), checkpoint directory of a run, or checkpoint file. By default the latest run.")
    parser.add_argument("--checkpoint", default="last", help="Checkpoint of the run (last, best, epoch_<e>), if run is not a file.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="The split of the run whose studies are evaluated.")
    parser.add_argument("--studies", nargs="+", default=None, help="Only these study ids of the split.")
    parser.add_argument("--max-studies", type=int, default=None, help="Only the first n studies of the split.")
    parser.add_argument("--data-root", type=Path, default=None, help="The simulated data, by default the run's.")
    parser.add_argument("--gpu", default="auto", help="'auto' for the first free GPU (see model.gpus.free_gpus), or a torch index.")
    parser.add_argument("--num-workers", type=int, default=2)


def latest_run(checkpoint_dir: Path) -> Path:
    """The run directory under checkpoint_dir with the latest run id (run ids start with their start time, see train/train.py) that has a
    checkpoint."""
    runs = sorted(d for d in checkpoint_dir.iterdir() if d.is_dir() and any(d.glob("*.pt")))
    assert runs, f"No runs with checkpoints under {checkpoint_dir}."
    return runs[-1]


@dataclass
class Setup:
    """A loaded checkpoint, its model and loss, and the loader of the studies to evaluate."""
    model: "HLPDModel"
    loss_fn: "HLPDLoss"
    zero_init: bool
    loader: DataLoader
    T: int
    shape: tuple[int, int, int]
    device: torch.device
    run_id: str
    checkpoint_path: Path
    epoch: int
    step: int
    out_dir: Path

    def header(self) -> dict:
        return {"run_id": self.run_id, "checkpoint": str(self.checkpoint_path), "epoch": self.epoch, "step": self.step}


def load_setup(args: argparse.Namespace, name: str) -> Setup:
    """Selects the GPU, loads the checkpoint of args (see add_common_args) into its model, and makes the loader of the studies to evaluate,
    with their volumes, and with the stored FBPs if the model is FBP initialized. The outputs go to
    <plot_dir>/<run_id>/<name>_<checkpoint>/<split>/."""
    from model.gpus import configure_cuda, select_gpu
    from train.evaluate import resolve_checkpoint

    configure_cuda()
    device = select_gpu(gpus=args.gpu)
    # NOTE: the model is imported after the GPU is selected, see train/train.py.
    from data.loaders import RegAndReconDataset, collate_time_list
    from model.build import HLPDConfig, build_hlpd, simulation_mu_water
    from model.hlpd import RayTrafoParams
    from train.train import study_shape

    run = args.run if args.run is not None else str(latest_run(args.checkpoint_dir))
    checkpoint_path = resolve_checkpoint(run, args.checkpoint, args.checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, run_id, epoch, step = checkpoint["config"], checkpoint["run_id"], checkpoint["epoch"], checkpoint["step"]
    logger.info("Checkpoint %s of run %s at epoch %d, step %d.", checkpoint_path, run_id, epoch, step)

    data_cfg = config["data"]
    data_root = args.data_root or Path(data_cfg["data_root"])
    qualities = data_cfg.get(f"{args.split}_qualities", data_cfg["val_qualities"])
    study_ids = list(data_cfg["study_ids"][args.split])
    assert study_ids, f"Run {run_id} has no {args.split} studies, choose another --split."
    if args.studies is not None:
        missing = set(args.studies) - set(study_ids)
        assert not missing, f"Studies {sorted(missing)} are not {args.split} studies of run {run_id}: {study_ids}."
        study_ids = args.studies
    if args.max_studies is not None:
        study_ids = sorted(study_ids)[:args.max_studies]

    model_config = HLPDConfig.from_dict(config)
    assert model_config.block.recurrent and not model_config.block.shared, \
        f"The tests compare the nets of the time bins, so they need a recurrent model with block.shared = False, but run {run_id} has {model_config.block}."
    dataset = RegAndReconDataset(qualities=qualities, mode=args.split, data_root=data_root, load_volume=True,
                                 load_fbp=not model_config.zero_init, study_ids=study_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_time_list, num_workers=args.num_workers)
    T, shape = study_shape(dataset)

    ray_cfg = config["ray_trafo"]
    ray_trafo_params = RayTrafoParams(**{**ray_cfg, "rotAxis": tuple(ray_cfg["rotAxis"])})
    model, loss_fn = build_hlpd(model_config, ray_trafo_params, T=T, shape=shape, device=device, mu_water=simulation_mu_water(data_root))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    del checkpoint

    out_dir = args.plot_dir / run_id / f"{name}_{checkpoint_path.stem}" / args.split
    logger.info("%d %s studies, outputs in %s.", len(dataset), args.split, out_dir)
    return Setup(model=model, loss_fn=loss_fn, zero_init=model_config.zero_init, loader=loader, T=T, shape=shape, device=device,
                 run_id=run_id, checkpoint_path=checkpoint_path, epoch=epoch, step=step, out_dir=out_dir)


# ----------------------------------------------------------------- data

def roll_bins(xs: Sequence, s: int) -> list:
    """The time bins cyclically shifted by s: bin t of the result is bin (t + s) mod T of xs."""
    T = len(xs)
    return [xs[(t + s) % T] for t in range(T)]


def simulate(model: "HLPDModel", x: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
    """Noisy data of the volumes x (T x (B, D, H, W) in mm^-1) with the model's current geometry (see set_geometry), as
    data/pipeline.simulate_study: bin t is projected with the geometry of bin t, and Poisson noise of the ray transform's flux is added.
    The noise is drawn from a generator seeded with seed, so that the data of different shifts only differ by the volumes, not by the
    seed. NOTE: the noise realization differs from the stored data's."""
    ray_trafo = model.ray_trafo
    torch.manual_seed(seed)
    with torch.no_grad():
        y = [ray_trafo.forward_bin(xt, t) for t, xt in enumerate(x)]
        return ray_trafo.add_poisson_noise(y)


# ----------------------------------------------------------------- net substitution

def unique_blocks(model: "HLPDModel") -> list["HLPDBlock"]:
    """The HLPD blocks of the iterations, each once (with unrolled.shared all iterations are the same block)."""
    blocks, seen = [], set()
    for block in model.unrolled_model.hlpd_model_list:
        if id(block) not in seen:
            seen.add(id(block))
            blocks.append(block)
    return blocks


# assign(k, group, nets) -> the nets of time bins 0..T-1 of group ("dual", "primal" or "motion") in block k, given its trained nets
Assignment = Callable[[int, str, nn.ModuleList], Sequence[nn.Module]]


@contextmanager
def assigned_nets(model: "HLPDModel", assign: Assignment, groups: Sequence[str] = NET_GROUPS) -> Iterator[None]:
    """Within the context, the nets of the given groups of every block are those of assign, e.g. rolled_nets or single_net. The trained nets
    are restored afterwards. The other groups keep their nets."""
    assert set(groups) <= set(NET_GROUPS), f"Unknown nets {sorted(set(groups) - set(NET_GROUPS))}. Expected some of {NET_GROUPS}."
    saved = []
    try:
        for k, block in enumerate(unique_blocks(model)):
            assert not block.shared, "The nets of the time bins are shared, there is nothing to substitute."
            for group in groups:
                nets = getattr(block, f"{group}_net")
                assert isinstance(nets, nn.ModuleList) and len(nets) == block.time_steps, f"Expected {block.time_steps} {group} nets, got {nets}."
                new = list(assign(k, group, nets))
                assert len(new) == len(nets), f"The assignment gives {len(new)} {group} nets for {len(nets)} time bins."
                saved.append((block, group, nets))
                setattr(block, f"{group}_net", nn.ModuleList(new))
        yield
    finally:
        for block, group, nets in reversed(saved):
            setattr(block, f"{group}_net", nets)


def rolled_nets(s: int) -> Assignment:
    """Time bin t gets the net of bin (t + s) mod T, i.e. the net that was trained on the phase that bin t has after roll_bins(x, s)."""
    return lambda k, group, nets: roll_bins(list(nets), s)


def single_net(j: int) -> Assignment:
    """All time bins get the net of bin j (of the same block and group)."""
    return lambda k, group, nets: [nets[j]] * len(nets)


def average_net(nets: Sequence[nn.Module]) -> nn.Module:
    """A copy of the nets with the mean of their weights. All nets of a group start from the same weights (HLPDBlock deep copies one
    template), so their average is a meaningful candidate for a net shared by all time bins."""
    states = [net.state_dict() for net in nets]
    mean = {name: torch.stack([state[name].float() for state in states]).mean(dim=0).to(states[0][name].dtype) for name in states[0]}
    net = copy.deepcopy(nets[0])
    net.load_state_dict(mean)
    return net


def averaged_nets() -> Assignment:
    """All time bins get the weight average of the nets of their block and group (see average_net)."""
    def assign(k: int, group: str, nets: nn.ModuleList) -> list[nn.Module]:
        return [average_net(list(nets))] * len(nets)
    return assign


def weight_distances(nets: Sequence[nn.Module]) -> np.ndarray:
    """The T x T relative distances ||theta_i - theta_j|| / rms(||theta_i||, ||theta_j||) of the flattened weights of the nets."""
    thetas = torch.stack([torch.cat([p.detach().float().flatten() for p in net.parameters()]) for net in nets])
    norms = thetas.norm(dim=1)
    distances = torch.cdist(thetas[None], thetas[None])[0]
    scale = torch.sqrt((norms[:, None] ** 2 + norms[None, :] ** 2) / 2).clamp_min(1e-12)
    return (distances / scale).cpu().numpy()


# ----------------------------------------------------------------- metrics

def _gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim3d(x: torch.Tensor, x_true: torch.Tensor, window: int = 7, sigma: float = 1.5, data_range: float | None = None) -> float:
    """The mean SSIM of the volumes x and x_true (B, D, H, W) with a separable Gaussian window, without padding, and the range of the
    ground truth as data range by default."""
    assert x.shape == x_true.shape and x.dim() == 4, f"Expected two (B, D, H, W) volumes, got {tuple(x.shape)} and {tuple(x_true.shape)}."
    if data_range is None:
        data_range = (x_true.max() - x_true.min()).item()
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    x, y = x.double().unsqueeze(1), x_true.double().unsqueeze(1)
    g = _gaussian_window(window, sigma, x.device, x.dtype)

    def blur(z: torch.Tensor) -> torch.Tensor:
        for dim in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[2 + dim] = window
            z = F.conv3d(z, g.view(shape))
        return z

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov = blur(x * y) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim.mean().item()


def psnr(x: torch.Tensor, x_true: torch.Tensor) -> float:
    """PSNR in dB with the range of the ground truth as peak, as train/evaluate.py."""
    mse = torch.mean((x - x_true) ** 2).item()
    peak = (x_true.max() - x_true.min()).item()
    return 10 * math.log10(peak ** 2 / mse) if mse > 0 else math.inf


def bin_metrics(w: list[torch.Tensor], f: list[torch.Tensor], x_true: list[torch.Tensor]) -> dict[str, list[float]]:
    """Per time bin: MSE, PSNR and SSIM of the reconstructions f, and MSE and PSNR of their predictions w by the motion (w_t = W(f_{t-1},
    v_{t-1}), so these measure the motion), all against the ground truth x_true."""
    return {
        "image_mse": [torch.mean((ft - xt) ** 2).item() for ft, xt in zip(f, x_true)],
        "image_psnr": [psnr(ft, xt) for ft, xt in zip(f, x_true)],
        "image_ssim": [ssim3d(ft, xt) for ft, xt in zip(f, x_true)],
        "warp_mse": [torch.mean((wt - xt) ** 2).item() for wt, xt in zip(w, x_true)],
        "warp_psnr": [psnr(wt, xt) for wt, xt in zip(w, x_true)],
    }


def with_means(metrics: dict[str, list[float]]) -> dict:
    """The per bin metrics and their means over the time bins (as mean_<metric>)."""
    return {**metrics, **{f"mean_{k}": float(np.mean(v)) for k, v in metrics.items()}}


def mean_over_studies(results: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """results[study][variant][metric] (per bin lists and means) -> mean[variant][metric], the mean over the studies, also per bin."""
    studies = list(results.values())
    mean = {}
    for variant, metrics in studies[0].items():
        mean[variant] = {}
        for k, val in metrics.items():
            values = [study[variant][k] for study in studies if variant in study]
            mean[variant][k] = np.mean(values, axis=0).tolist() if isinstance(val, list) else float(np.mean(values))
    return mean


def forward(setup: Setup, g: list[torch.Tensor], fbp: list[torch.Tensor] | None) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """(w, f, v) of the model for the data g. fbp is the stored FBP of g, or None to compute it (only used with FBP initialization)."""
    with torch.no_grad():
        return setup.model(g, None if setup.zero_init else fbp)


# ----------------------------------------------------------------- plotting (see the dataviz reference palette)

COLORS = {"series1": "#2a78d6", "series2": "#eb6834", "series3": "#1baf7a", "muted": "#8a8984", "text": "#0b0b0b",
          "text_secondary": "#52514e", "grid": "#e4e3df", "neutral": "#e8e7e3"}


def style_axes(ax) -> None:
    """Recessive grid and axes."""
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)


def diverging_cmap():
    """Orange (worse) - neutral gray - blue (better), for metric differences to a baseline."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("worse_better", [COLORS["series2"], COLORS["neutral"], COLORS["series1"]])


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("magnitude", ["#f3f6fb", COLORS["series1"], "#0d3466"])
