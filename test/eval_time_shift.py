"""
Equivariance of a trained HLPD model with per time bin nets (block.shared = False) under cyclic shifts of the phases, the action of Z/T on
the T time bins.

For each shift s, the ground truth phases are rolled, x_s[t] = x[(t + s) mod T], and projected anew with the unchanged geometry of each bin
(test/time_bin_perturbation.simulate). Only the phases change, not the gantry arcs: bin t covers the arc [t, t + 1] * GantrySpeed (see
operators/tomo/ray_trafo.py), so rolling the stored sinograms would also hand each net the limited angle artifacts of another arc. Each shift
is reconstructed by
    shift_s<s>        the trained model: the net of bin t now sees phase (t + s) mod T.
    shift_s<s>_nets   the nets rolled along with the phases (bin t uses the net of bin (t + s) mod T), so every net sees the phase it was
                      trained on, but an arc and a position in the recurrent sweep it was not trained on. NOTE: the sweep starts at bin 0,
                      whose previous time bin is the last bin of the previous iteration, while all other bins get their predecessor of the
                      current iteration (see HLPDBlock.forward), so even shared nets would not be exactly equivariant.
The s = 0 reconstruction of the re-simulated data is the baseline (all shifts use the same noise seed), and the stored data (another noise
realization) is reconstructed as well, as a check that the re-simulation matches the data the model was trained on.

Metrics (test/time_bin_perturbation.bin_metrics), per time bin and their means: MSE, PSNR and SSIM of the reconstructions, and MSE and PSNR
of their predictions by the motion, against the ground truth of the shifted phases. And the equivariance errors, against the baseline
reconstruction of the same phase p = (t + s) mod T:
    equivariance_ratio   ||f_s[t] - f_0[p]|| / ||f_0[p] - x[p]||, the change of the reconstruction relative to the baseline's error
    velocity_rel_diff    ||v_s[t] - v_0[p]|| / ||v_0[p]||
An equivariant model has the same metrics for all shifts and zero equivariance errors.

Run from the repo root, e.g. on the last checkpoint of the latest run:
    python -m test.eval_time_shift
    python -m test.eval_time_shift 20261105-130953_hlpd_train --shifts 0 2 5 --max-studies 2

Writes to <plot-dir>/<run_id>/time_shift_<checkpoint>/<split>/:
    metrics.json                the metrics of each study and variant, and their means over the studies (per bin and per phase)
    shift_metrics.png           PSNR and SSIM of the reconstructions and PSNR of the motion predictions against the shift
    shift_equivariance.png      the equivariance errors against the shift
    shift_phase_psnr.png        the PSNR change to the baseline per shift and phase
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

from test.time_bin_perturbation import (COLORS, add_common_args, assigned_nets, bin_metrics, diverging_cmap, forward, load_setup,
                                        mean_over_studies, roll_bins, rolled_nets, simulate, style_axes, with_means)

logger = logging.getLogger(__name__)

METRICS = ("image_mse", "image_psnr", "image_ssim", "warp_mse", "warp_psnr", "equivariance_ratio", "velocity_rel_diff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--shifts", type=int, nargs="+", default=None, help="The shifts to evaluate, by default 0, ..., T-1. 0 is always included.")
    parser.add_argument("--no-rolled-nets", action="store_true", help="Skip the reconstructions with the nets rolled along with the phases.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the noise of the re-simulated data, the same for all shifts.")
    return parser.parse_args()


def equivariance_errors(f: list[torch.Tensor], v: list[torch.Tensor], f0: list[torch.Tensor], v0: list[torch.Tensor],
                        x_true: list[torch.Tensor], s: int) -> dict[str, list[float]]:
    """The equivariance errors (see the module docstring) of the reconstruction (f, v) of the data shifted by s, per time bin t, against the
    baseline (f0, v0) of the unshifted data with ground truth x_true."""
    f0_s, v0_s, x_s = roll_bins(f0, s), roll_bins(v0, s), roll_bins(x_true, s)
    return {
        "equivariance_ratio": [((ft - bt).norm() / (bt - xt).norm().clamp_min(1e-12)).item() for ft, bt, xt in zip(f, f0_s, x_s)],
        "velocity_rel_diff": [((vt - bt).norm() / bt.norm().clamp_min(1e-12)).item() for vt, bt in zip(v, v0_s)],
    }


def by_phase(values: list[float], s: int) -> list[float]:
    """Per bin values of the data shifted by s, reordered by phase: bin t holds phase (t + s) mod T."""
    T = len(values)
    return [values[(p - s) % T] for p in range(T)]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from model.build import set_geometry
    from train.train import to_device

    setup = load_setup(args, "time_shift")
    T, model = setup.T, setup.model
    shifts = sorted({0, *(s % T for s in (args.shifts if args.shifts is not None else range(T)))})
    logger.info("Shifts %s%s.", shifts, "" if args.no_rolled_nets else ", with and without rolling the nets")

    results: dict[str, dict] = {}
    for batch in setup.loader:
        study_id = batch["study_id"][0]
        g_stored, x_true, fbp_stored, meta = to_device(batch, setup.device)
        set_geometry(model, setup.loss_fn, meta)
        study: dict[str, dict] = {}

        w, f, v = forward(setup, g_stored, fbp_stored)
        study["stored"] = with_means(bin_metrics(w, f, x_true)) | {"shift": 0}
        del g_stored, fbp_stored, w, f, v

        f0 = v0 = None
        for s in shifts:
            x_s = roll_bins(x_true, s)
            g_s = simulate(model, x_s, args.seed)
            variants = [(f"shift_s{s}", None)] + ([(f"shift_s{s}_nets", rolled_nets(s))] if s != 0 and not args.no_rolled_nets else [])
            for name, assignment in variants:
                if assignment is None:
                    w, f, v = forward(setup, g_s, None)
                else:
                    with assigned_nets(model, assignment):
                        w, f, v = forward(setup, g_s, None)
                if s == 0:
                    f0, v0 = f, v
                assert f0 is not None and v0 is not None
                metrics = bin_metrics(w, f, x_s) | equivariance_errors(f, v, f0, v0, x_true, s)
                study[name] = with_means(metrics) | {"shift": s}
                logger.info("%s %s: PSNR %.3f dB, SSIM %.4f, warp PSNR %.3f dB, equivariance ratio %.3g, velocity rel. diff %.3g", study_id, name,
                            *(study[name][f"mean_{k}"] for k in ("image_psnr", "image_ssim", "warp_psnr", "equivariance_ratio", "velocity_rel_diff")))
            del g_s, x_s
        results[study_id] = study
        del x_true, f0, v0

    mean = mean_over_studies(results)
    # Per phase means, to compare the same phase across shifts (bin t of shift s holds phase (t + s) mod T).
    for variant in mean.values():
        for k in METRICS:
            if k in variant:
                variant[f"{k}_per_phase"] = by_phase(variant[k], int(variant["shift"]))
    logger.info("Mean over %d studies:", len(results))
    for name, metrics in mean.items():
        logger.info("  %-16s PSNR %.3f dB, SSIM %.4f, warp PSNR %.3f dB, equivariance ratio %.3g, velocity rel. diff %.3g", name,
                    *(metrics.get(f"mean_{k}", float("nan")) for k in ("image_psnr", "image_ssim", "warp_psnr", "equivariance_ratio", "velocity_rel_diff")))

    setup.out_dir.mkdir(parents=True, exist_ok=True)
    with open(setup.out_dir / "metrics.json", "w") as f_out:
        json.dump(setup.header() | {"split": args.split, "shifts": shifts, "seed": args.seed, "mean": mean, "studies": results}, f_out, indent=2)
    logger.info("Wrote %s.", setup.out_dir / "metrics.json")
    plot(mean, shifts, T, setup.out_dir, f"{setup.run_id}, epoch {setup.epoch}, {len(results)} {args.split} studies")


# ----------------------------------------------------------------- plots

def _series(mean: dict, shifts: list[int], key: str, nets: bool) -> tuple[list[int], list[float]]:
    """(shifts, mean metric) of the plain shifts, or of the shifts with rolled nets (whose s = 0 is the baseline)."""
    xs, ys = [], []
    for s in shifts:
        name = f"shift_s{s}_nets" if nets and s != 0 else f"shift_s{s}"
        if name in mean:
            xs.append(s)
            ys.append(mean[name][f"mean_{key}"])
    return xs, ys


def _line_panels(mean: dict, shifts: list[int], panels: list[tuple[str, str]], path: Path, title: str, baseline: bool) -> None:
    has_nets = any(name.endswith("_nets") for name in mean)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), constrained_layout=True)
    for ax, (key, label) in zip(np.atleast_1d(axes), panels):
        style_axes(ax)
        if baseline:
            ax.axhline(mean["shift_s0"][f"mean_{key}"], color=COLORS["muted"], linestyle="--", linewidth=1, label="baseline (s = 0)")
        xs, ys = _series(mean, shifts, key, nets=False)
        ax.plot(xs, ys, color=COLORS["series1"], linewidth=2, marker="o", markersize=5, label="phases shifted")
        if has_nets:
            xs, ys = _series(mean, shifts, key, nets=True)
            ax.plot(xs, ys, color=COLORS["series2"], linewidth=2, marker="s", markersize=5, label="phases and nets shifted")
        ax.set_xticks(shifts)
        ax.set_xlabel("shift s (time bins)", color=COLORS["text_secondary"], fontsize=9)
        ax.set_title(label, color=COLORS["text"], fontsize=10, loc="left")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=COLORS["text_secondary"], fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot(mean: dict, shifts: list[int], T: int, out_dir: Path, title: str) -> None:
    _line_panels(mean, shifts, [("image_psnr", "Reconstruction PSNR (dB)"), ("image_ssim", "Reconstruction SSIM"),
                                ("warp_psnr", "Motion prediction PSNR (dB)")], out_dir / "shift_metrics.png", title, baseline=True)
    _line_panels(mean, shifts, [("equivariance_ratio", "||f_s - f_0|| / ||f_0 - x|| (same phase)"),
                                ("velocity_rel_diff", "||v_s - v_0|| / ||v_0|| (same phase)")], out_dir / "shift_equivariance.png", title,
                 baseline=False)

    # The PSNR change to the baseline per shift and phase.
    base = np.array(mean["shift_s0"]["image_psnr_per_phase"])
    panels = [("phases shifted", "")] + ([("phases and nets shifted", "_nets")] if any(n.endswith("_nets") for n in mean) else [])
    deltas = {label: np.array([np.array(mean[f"shift_s{s}{suffix}" if s != 0 else "shift_s0"]["image_psnr_per_phase"]) - base for s in shifts])
              for label, suffix in panels}
    limit = max(float(np.abs(d).max()) for d in deltas.values()) or 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 0.32 * len(shifts) + 1.8), constrained_layout=True, squeeze=False)
    for ax, (label, delta) in zip(axes[0], deltas.items()):
        im = ax.imshow(delta, cmap=diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(T))
        ax.set_yticks(range(len(shifts)), [str(s) for s in shifts])
        ax.set_xlabel("phase", color=COLORS["text_secondary"], fontsize=9)
        ax.set_ylabel("shift s", color=COLORS["text_secondary"], fontsize=9)
        ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
        ax.set_title(f"PSNR change to the baseline (dB), {label}", color=COLORS["text"], fontsize=10, loc="left")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, label="dB (blue better, orange worse)")
    fig.suptitle(title, color=COLORS["text_secondary"], fontsize=9)
    fig.savefig(out_dir / "shift_phase_psnr.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote the plots to %s.", out_dir)


if __name__ == "__main__":
    main()
