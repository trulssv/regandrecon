"""
How much the per time bin nets of a trained HLPD model (block.shared = False) have specialized: the nets of all time bins are replaced by
the net of a single bin j, in every iteration (block) separately, i.e. bin t of iteration k uses the net of bin j of iteration k. Variants:
    trained     the trained model
    net_<j>     all time bins use the nets of bin j
    average     all time bins use the weight average of the nets of their iteration (test/time_bin_perturbation.average_net). The nets of
                a group start from the same weights, so the average is a candidate for a net shared by all time bins.
--groups selects which nets are replaced (dual, primal, motion; by default all), e.g. --groups motion to see whether the specialization
is in the motion.

NOTE: this measures how far the trained nets have specialized, not how well a shared model could do, as no single net was trained for all
time bins. The net of bin 0 is moreover the only one trained with the previous iteration's last bin as predecessor (see HLPDBlock.forward).

Metrics (test/time_bin_perturbation.bin_metrics) on the stored data, per time bin and their means: MSE, PSNR and SSIM of the
reconstructions, and MSE and PSNR of their predictions by the motion. Also the relative distances between the weights of the nets of each
iteration and group (test/time_bin_perturbation.weight_distances), which need no data.

Run from the repo root, e.g. on the last checkpoint of the latest run:
    python -m test.eval_net_substitution
    python -m test.eval_net_substitution 20261105-130953_hlpd_train --donors 0 5 --groups motion --max-studies 2

Writes to <plot-dir>/<run_id>/net_substitution_<groups>_<checkpoint>/<split>/:
    metrics.json                    the metrics of each study and variant, their means over the studies, and the weight distances
    substitution_metrics.png        PSNR and SSIM of the reconstructions and PSNR of the motion predictions of each variant
    substitution_bin_psnr.png       the PSNR change to the trained model per variant and time bin
    weight_distances.png            the relative weight distances between the nets of the time bins, per iteration and group
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from test.time_bin_perturbation import (COLORS, NET_GROUPS, add_common_args, assigned_nets, averaged_nets, bin_metrics, diverging_cmap,
                                        forward, load_setup, mean_over_studies, sequential_cmap, single_net, style_axes, unique_blocks,
                                        weight_distances, with_means)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--groups", nargs="+", default=list(NET_GROUPS), choices=NET_GROUPS, help="The nets that are replaced.")
    parser.add_argument("--donors", type=int, nargs="+", default=None, help="The time bins whose nets replace all others, by default all.")
    parser.add_argument("--no-average", action="store_true", help="Skip the weight averaged nets.")
    return parser.parse_args()


def all_weight_distances(model, groups: list[str]) -> dict[str, list]:
    """The T x T relative weight distances of the nets of each group, per iteration (block): distances[group][k][i][j]."""
    return {group: [weight_distances(list(getattr(block, f"{group}_net"))).tolist() for block in unique_blocks(model)] for group in groups}


def mean_off_diagonal(d: np.ndarray) -> float:
    T = d.shape[0]
    return float(d[~np.eye(T, dtype=bool)].mean()) if T > 1 else 0.0


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from model.build import set_geometry
    from train.train import to_device

    groups = [g for g in NET_GROUPS if g in args.groups]
    setup = load_setup(args, f"net_substitution_{'-'.join(groups)}")
    T, model = setup.T, setup.model
    donors = sorted({j % T for j in (args.donors if args.donors is not None else range(T))})

    distances = all_weight_distances(model, list(NET_GROUPS))
    for group in NET_GROUPS:
        logger.info("Mean relative weight distance between the %s nets of the time bins, per iteration: %s", group,
                    ", ".join(f"{mean_off_diagonal(np.array(d)):.3g}" for d in distances[group]))

    variants = [("trained", None)] + [(f"net_{j}", single_net(j)) for j in donors] + ([] if args.no_average else [("average", averaged_nets())])
    logger.info("Replacing the %s nets: %s.", ", ".join(groups), ", ".join(name for name, _ in variants[1:]))

    results: dict[str, dict] = {}
    for batch in setup.loader:
        study_id = batch["study_id"][0]
        g, x_true, fbp, meta = to_device(batch, setup.device)
        set_geometry(model, setup.loss_fn, meta)
        study: dict[str, dict] = {}
        for name, assignment in variants:
            if assignment is None:
                w, f, _ = forward(setup, g, fbp)
            else:
                with assigned_nets(model, assignment, groups):
                    w, f, _ = forward(setup, g, fbp)
            study[name] = with_means(bin_metrics(w, f, x_true))
            logger.info("%s %s: PSNR %.3f dB, SSIM %.4f, warp PSNR %.3f dB", study_id, name,
                        *(study[name][f"mean_{k}"] for k in ("image_psnr", "image_ssim", "warp_psnr")))
            del w, f
        results[study_id] = study
        del g, x_true, fbp

    mean = mean_over_studies(results)
    logger.info("Mean over %d studies:", len(results))
    for name, metrics in mean.items():
        logger.info("  %-10s PSNR %.3f dB, SSIM %.4f, warp PSNR %.3f dB", name, metrics["mean_image_psnr"], metrics["mean_image_ssim"],
                    metrics["mean_warp_psnr"])

    setup.out_dir.mkdir(parents=True, exist_ok=True)
    with open(setup.out_dir / "metrics.json", "w") as f_out:
        json.dump(setup.header() | {"split": args.split, "groups": groups, "donors": donors, "mean": mean, "studies": results,
                                    "weight_distances": distances}, f_out, indent=2)
    logger.info("Wrote %s.", setup.out_dir / "metrics.json")
    title = f"{setup.run_id}, epoch {setup.epoch}, {len(results)} {args.split} studies, {', '.join(groups)} nets replaced"
    plot(mean, donors, T, distances, setup.out_dir, title)


# ----------------------------------------------------------------- plots

def plot(mean: dict, donors: list[int], T: int, distances: dict[str, list], out_dir: Path, title: str) -> None:
    # Mean metrics of each variant.
    panels = [("image_psnr", "Reconstruction PSNR (dB)"), ("image_ssim", "Reconstruction SSIM"), ("warp_psnr", "Motion prediction PSNR (dB)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), constrained_layout=True)
    for ax, (key, label) in zip(axes, panels):
        style_axes(ax)
        ax.axhline(mean["trained"][f"mean_{key}"], color=COLORS["muted"], linestyle="--", linewidth=1, label="trained")
        if "average" in mean:
            ax.axhline(mean["average"][f"mean_{key}"], color=COLORS["series2"], linewidth=2, label="weight average")
        ax.plot(donors, [mean[f"net_{j}"][f"mean_{key}"] for j in donors], color=COLORS["series1"], linestyle="none", marker="o",
                markersize=7, label="net of bin j for all bins")
        ax.set_xticks(donors)
        ax.set_xlabel("donor time bin j", color=COLORS["text_secondary"], fontsize=9)
        ax.set_title(label, color=COLORS["text"], fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=COLORS["text_secondary"], fontsize=9)
    fig.savefig(out_dir / "substitution_metrics.png", dpi=150)
    plt.close(fig)

    # The PSNR change to the trained model per variant and time bin.
    names = [f"net_{j}" for j in donors] + (["average"] if "average" in mean else [])
    base = np.array(mean["trained"]["image_psnr"])
    delta = np.array([np.array(mean[name]["image_psnr"]) - base for name in names])
    limit = float(np.abs(delta).max()) or 1.0
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(names) + 1.8), constrained_layout=True)
    im = ax.imshow(delta, cmap=diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(T))
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("time bin t", color=COLORS["text_secondary"], fontsize=9)
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    ax.set_title("PSNR change to the trained model (dB)", color=COLORS["text"], fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, label="dB (blue better, orange worse)")
    fig.suptitle(title, color=COLORS["text_secondary"], fontsize=8)
    fig.savefig(out_dir / "substitution_bin_psnr.png", dpi=150)
    plt.close(fig)

    # The weight distances, per group (rows) and iteration (columns), on one scale.
    groups = list(distances)
    K = len(distances[groups[0]])
    vmax = max(float(np.max(d)) for group in groups for d in distances[group]) or 1.0
    fig, axes = plt.subplots(len(groups), K, figsize=(1.9 * K + 1.2, 1.9 * len(groups) + 0.8), constrained_layout=True, squeeze=False)
    for r, group in enumerate(groups):
        for k in range(K):
            ax = axes[r, k]
            d = np.array(distances[group][k])
            im = ax.imshow(d, cmap=sequential_cmap(), vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{group}, iter {k}: {mean_off_diagonal(d):.2g}", color=COLORS["text"], fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="||θ_i - θ_j|| / rms(||θ_i||, ||θ_j||)")
    fig.suptitle("Relative weight distances between the nets of time bins i, j (title: mean off the diagonal)", color=COLORS["text_secondary"],
                 fontsize=9)
    fig.savefig(out_dir / "weight_distances.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote the plots to %s.", out_dir)


if __name__ == "__main__":
    main()
