"""
Evaluates a checkpoint of a training run (train/train.py) on the studies of one of its splits, by default the test studies. The model,
the ray transform and the study ids of the splits are taken from the checkpoint's config, so only the checkpoint is needed.

Run from the repo root, e.g. on the latest checkpoint of a run:
    python -m train.evaluate 20261026-141651_hlpd_train
The checkpoint is a run id (under --checkpoint-dir), a run's checkpoint directory, or a checkpoint file. For a run, --checkpoint picks the
file: last (default), best or e.g. epoch_0119.

Writes to <plot-dir>/<run_id>/eval_<checkpoint>/<split>/, next to the run's other plots:
    metrics.json                                    the metrics of each study, and their mean
    <study_id>/prediction.gif                       ground truth, reconstruction, motion prediction and RGB velocity over the time bins
    <study_id>/velocity_quiver.gif                  the in-plane velocity as arrows over the time bins
    <study_id>/reconstruction_t<t>.png              ground truth, FBP, reconstruction, its error, motion prediction and RGB velocity of time bin t
    <study_id>/velocity_quiver_t<t>.png             the in-plane velocity of time bin t as arrows
all in the axial, coronal and sagittal mid planes, and
    <study_id>/sinogram.gif                         the data, the reprojection of the reconstruction and their difference over the time bins
    <study_id>/sinogram_t<t>.png                    the same of time bin t
in the middle projection and the sinograms of the middle detector row and column (see tracking/evaluation_plots.py).
The arrows and colors of the quiver plots have the same scale in all studies, the other plots have the scales of their study.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.loaders import RegAndReconDataset, collate_time_list
from model.gpus import configure_cuda, select_gpu
from tracking.evaluation_plots import (in_plane_speeds, plot_reconstruction, plot_sinogram, plot_sinogram_gif, plot_velocity_quiver,
                                       plot_velocity_quiver_gif, quiver_scales, sinogram_scales)
from tracking.training_plots import plot_prediction_gif
from train.train import step_metrics, study_shape, to_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Run id (under --checkpoint-dir), checkpoint directory of a run, or checkpoint file.")
    parser.add_argument("--checkpoint", default="last", help="Checkpoint of the run to evaluate (last, best, epoch_<e>), if run is not a file.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/media/trulssv/LDDMM/checkpoints"))
    parser.add_argument("--plot-dir", type=Path, default=Path("test/plots"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="The split of the run whose studies are evaluated.")
    parser.add_argument("--studies", nargs="+", default=None, help="Only these study ids of the split.")
    parser.add_argument("--data-root", type=Path, default=None, help="The simulated data, by default the run's.")
    parser.add_argument("--time-bins", type=int, nargs="+", default=None, help="The time bins of the PNGs (0-based), by default all.")
    parser.add_argument("--n-arrows", type=int, default=24, help="Number of arrows along the longest side of the quiver plots.")
    parser.add_argument("--gpu", default="auto", help="'auto' for the first free GPU (see model.gpus.free_gpus), or a torch index.")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def resolve_checkpoint(run: str, checkpoint: str, checkpoint_dir: Path) -> Path:
    path = Path(run)
    if path.is_file():
        return path
    run_dir = path if path.is_dir() else checkpoint_dir / run
    path = run_dir / (checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint {path}, found {sorted(p.name for p in run_dir.glob('*.pt'))}.")
    return path


def psnr(x: torch.Tensor, x_true: torch.Tensor) -> float:
    """PSNR in dB with the range of the ground truth as peak."""
    mse = torch.mean((x - x_true) ** 2).item()
    peak = (x_true.max() - x_true.min()).item()
    return 10 * math.log10(peak ** 2 / mse) if mse > 0 else math.inf


def study_metrics(loss, w: list[torch.Tensor], f: list[torch.Tensor], x_true: list[torch.Tensor], fbp: list[torch.Tensor] | None) -> dict:
    """The loss terms and image domain errors of train.train.step_metrics, the PSNR of the reconstructions, and the error and PSNR of the
    FBPs and of the zero prediction for reference, also per time bin."""
    metrics = step_metrics(loss, w, f, x_true)
    metrics["image_psnr"] = float(np.mean([psnr(ft, xt) for ft, xt in zip(f, x_true)]))
    metrics["zero_prediction_image_mse"] = torch.stack([xt.square().mean() for xt in x_true]).mean().item()
    metrics["image_mse_per_bin"] = [torch.mean((ft - xt) ** 2).item() for ft, xt in zip(f, x_true)]
    if fbp is not None:
        metrics["fbp_image_mse"] = torch.stack([torch.mean((bt - xt) ** 2) for bt, xt in zip(fbp, x_true)]).mean().item()
        metrics["fbp_image_psnr"] = float(np.mean([psnr(bt, xt) for bt, xt in zip(fbp, x_true)]))
        metrics["fbp_image_mse_per_bin"] = [torch.mean((bt - xt) ** 2).item() for bt, xt in zip(fbp, x_true)]
    return metrics


def to_numpy(xs: list[torch.Tensor], shape: tuple[int, ...]) -> list[np.ndarray]:
    """The first batch element of each tensor as a numpy array of the given shape."""
    return [x[0].float().cpu().numpy().reshape(shape) for x in xs]


@dataclass
class StudyOutput:
    """What the plots need of a study, as numpy arrays over the time bins (see tracking/evaluation_plots.py)."""
    label: str
    extent: tuple[float, float, float]
    x_true: list[np.ndarray]
    f: list[np.ndarray]
    w: list[np.ndarray]
    v: list[np.ndarray]
    fbp: list[np.ndarray] | None
    g: list[np.ndarray]
    reprojection: list[np.ndarray]
    angles: list[tuple[float, float]]


def plot_study(study_dir: Path, out: StudyOutput, detector_extent: tuple[float, float], time_bins: list[int], n_arrows: int,
               gain: float, speed_max: float) -> None:
    """The plots of tracking/evaluation_plots.py of one study, with the scales of the whole study so that the time bins compare, and
    the given quiver scales (see quiver_scales)."""
    vmin, vmax = min(x.min() for x in out.x_true), max(x.max() for x in out.x_true)
    v_scale = float(np.percentile(np.abs(np.stack(out.v)), 99)) or 1.0
    sino_scales = sinogram_scales(out.g, out.reprojection)
    T = len(out.x_true)
    for t in time_bins:
        title = f"{out.label}, time bin {t + 1}/{T}"
        plot_reconstruction(out.x_true[t], out.f[t], out.v[t], out.extent, study_dir / f"reconstruction_t{t:02d}.png", title, x_warped=out.w[t],
                            fbp=out.fbp[t] if out.fbp is not None else None, v_scale=v_scale, vmin=vmin, vmax=vmax)
        plot_velocity_quiver(out.f[t], out.v[t], out.extent, study_dir / f"velocity_quiver_t{t:02d}.png", title, n_arrows=n_arrows, gain=gain,
                             vmin=vmin, vmax=vmax, speed_max=speed_max)
        plot_sinogram(out.g, out.angles, detector_extent, t, study_dir / f"sinogram_t{t:02d}.png", title, reprojection=out.reprojection,
                      scales=sino_scales)
    plot_velocity_quiver_gif(out.f, out.v, out.extent, study_dir / "velocity_quiver.gif", out.label, n_arrows=n_arrows, vmin=vmin, vmax=vmax,
                             gain=gain, speed_max=speed_max)
    plot_sinogram_gif(out.g, out.angles, detector_extent, study_dir / "sinogram.gif", out.label, reprojection=out.reprojection, scales=sino_scales)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_cuda()
    device = select_gpu(gpus=args.gpu)
    # NOTE: the model is imported after the GPU is selected, see train/train.py.
    from model.build import HLPDConfig, build_hlpd, set_geometry, simulation_mu_water
    from model.hlpd import RayTrafoParams

    checkpoint_path = resolve_checkpoint(args.run, args.checkpoint, args.checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, run_id, epoch, step = checkpoint["config"], checkpoint["run_id"], checkpoint["epoch"], checkpoint["step"]
    logger.info("Checkpoint %s of run %s at epoch %d, step %d.", checkpoint_path, run_id, epoch, step)

    data_cfg = config["data"]
    data_root = args.data_root or Path(data_cfg["data_root"])
    qualities = data_cfg.get(f"{args.split}_qualities", data_cfg["val_qualities"])
    study_ids = data_cfg["study_ids"][args.split]
    if args.studies is not None:
        missing = set(args.studies) - set(study_ids)
        assert not missing, f"Studies {sorted(missing)} are not {args.split} studies of run {run_id}: {study_ids}."
        study_ids = args.studies
    model_config = HLPDConfig.from_dict(config)
    # NOTE: the FBP is loaded also with zero initialization, as the reference of the reconstructions.
    dataset = RegAndReconDataset(qualities=qualities, mode=args.split, data_root=data_root, load_volume=True, load_fbp=True, study_ids=study_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_time_list, num_workers=args.num_workers)
    T, shape = study_shape(dataset)

    ray_cfg = config["ray_trafo"]
    ray_trafo_params = RayTrafoParams(**{**ray_cfg, "rotAxis": tuple(ray_cfg["rotAxis"])})
    model, loss_fn = build_hlpd(model_config, ray_trafo_params, T=T, shape=shape, device=device, mu_water=simulation_mu_water(data_root))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    del checkpoint

    out_dir = args.plot_dir / run_id / f"eval_{checkpoint_path.stem}" / args.split
    time_bins = args.time_bins if args.time_bins is not None else list(range(T))
    logger.info("Evaluating %d %s studies, plots in %s.", len(dataset), args.split, out_dir)

    # The plots wait until all studies are evaluated, so that the quiver plots of all studies get the same scales.
    results, outputs = {}, {}
    for batch in loader:
        study_id = batch["study_id"][0]
        g, x_true, fbp, meta = to_device(batch, device)
        set_geometry(model, loss_fn, meta)
        w, f, v = model(g, fbp if not model_config.zero_init else None)
        loss = loss_fn(w, f, v, g if model_config.loss.domain == "projection" else x_true)
        results[study_id] = study_metrics(loss, w, f, x_true, fbp)
        logger.info("%s: %s", study_id, ", ".join(f"{k}={val:.4g}" for k, val in results[study_id].items() if not isinstance(val, list)))

        plot_prediction_gif(f, x_true, v, model.extent, out_dir / study_id / "prediction.gif", step, x_warped=w, epoch=epoch)
        sino_shape = tuple(g[0].shape[1:])
        reprojection = [model.ray_trafo.forward_bin(ft, t) for t, ft in enumerate(f)]
        angles = [(float(d.min_pt[0]), float(d.max_pt[0])) for d in model.ray_trafo.angular_discretizations]
        outputs[study_id] = StudyOutput(
            label=f"{study_id} ({run_id}, epoch {epoch})", extent=tuple(float(e) for e in model.extent), x_true=to_numpy(x_true, shape),
            f=to_numpy(f, shape), w=to_numpy(w, shape), v=to_numpy(v, (*shape, 3)), fbp=to_numpy(fbp, shape) if fbp is not None else None,
            g=to_numpy(g, sino_shape), reprojection=to_numpy(reprojection, sino_shape), angles=angles)
        del g, x_true, fbp, w, f, v, loss, reprojection

    speeds = np.concatenate([in_plane_speeds(np.stack(out.v), shape, out.extent, args.n_arrows) for out in outputs.values()])
    gain, speed_max = quiver_scales(speeds, max(outputs.values(), key=lambda out: max(out.extent)).extent, args.n_arrows)
    detector_extent = (ray_trafo_params.DetectorColExtent, ray_trafo_params.DetectorRowExtent)
    logger.info("Plotting, quiver arrows %.3g x the velocity, colors up to %.3g mm.", gain, speed_max)
    for study_id, out in outputs.items():
        plot_study(out_dir / study_id, out, detector_extent, time_bins, args.n_arrows, gain, speed_max)

    first = next(iter(results.values()))
    mean = {k: float(np.mean([r[k] for r in results.values()])) for k, val in first.items() if not isinstance(val, list)}
    logger.info("Mean over %d studies: %s", len(results), ", ".join(f"{k}={val:.4g}" for k, val in mean.items()))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f_out:
        json.dump({"run_id": run_id, "checkpoint": str(checkpoint_path), "epoch": epoch, "step": step, "split": args.split,
                   "mean": mean, "studies": results}, f_out, indent=2)
    logger.info("Wrote %s.", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
