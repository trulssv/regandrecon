"""
Plots of the evaluation of a trained model on a study (see train/evaluate.py), all in the axial, coronal and sagittal mid planes:
    reconstruction_t<t>.png     ground truth, FBP (if given), reconstruction, its error, motion prediction and velocity (RGB) of time bin t
    velocity_quiver_t<t>.png    the in-plane velocity of time bin t as arrows on top of the reconstruction
    velocity_quiver.gif         the same over the time bins
and in the mid slices of the sinograms:
    sinogram_t<t>.png           the data, the reprojection of the reconstruction and their difference of time bin t
    sinogram.gif                the same over the time bins
The GIF of the ground truth, reconstruction, motion prediction and RGB velocity over the time bins is tracking.training_plots.plot_prediction_gif.

Volumes are (D, H, W) numpy arrays and velocities (D, H, W, 3) with grid_sample's (x, y, z) = (W, H, D) channel order in mm (see
FlowDeformationOperator), so the velocity component along volume axis a is channel 2 - a. Sinograms are (views, cols, rows) numpy
arrays of line integrals, with the angular range (in radians) of their time bin and the (col, row) extent of the detector in mm.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from tracking.training_plots import PLANES, _mid_slice, _save_atomic


def _imshow(ax, image: np.ndarray, plane: str, extent: tuple[float, float, float], **kwargs):
    """image (a mid plane) with its physical extent, as in plot_prediction_gif."""
    _, (row_axis, col_axis) = PLANES[plane]
    im = ax.imshow(image, origin="lower", extent=(0, extent[col_axis], 0, extent[row_axis]), **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def velocity_rgb(vt: np.ndarray, plane: str, v_scale: float) -> np.ndarray:
    """The mid plane of the velocity as RGB = (x, y, z), mapped symmetrically around gray (zero velocity) and saturated at ±v_scale."""
    return np.clip(0.5 + 0.5 * _mid_slice(vt, plane) / v_scale, 0.0, 1.0)


def plot_reconstruction(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    v: np.ndarray,
    extent: tuple[float, float, float],
    path: Path,
    title: str,
    x_warped: np.ndarray | None = None,
    fbp: np.ndarray | None = None,
    v_scale: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Ground truth, FBP (if given), reconstruction, its absolute error, motion prediction (if given) and RGB velocity of one time bin
    in the three mid planes. The gray images share vmin/vmax (by default the range of the ground truth), and the velocity is
    saturated at ±v_scale (by default its 99th percentile), so that pass the values of the whole study to compare time bins."""
    vmin = float(x_true.min()) if vmin is None else vmin
    vmax = float(x_true.max()) if vmax is None else vmax
    v_scale = v_scale or float(np.percentile(np.abs(v), 99)) or 1.0
    error = np.abs(x_pred - x_true)
    error_max = float(np.percentile(error, 99.9)) or 1.0

    gray = {"cmap": "gray", "vmin": vmin, "vmax": vmax}
    rows = [("ground truth", lambda p: _mid_slice(x_true, p), gray)]
    if fbp is not None:
        rows.append(("FBP", lambda p: _mid_slice(fbp, p), gray))
    rows.append(("reconstruction", lambda p: _mid_slice(x_pred, p), gray))
    rows.append(("|reconstruction - truth|", lambda p: _mid_slice(error, p), {"cmap": "magma", "vmin": 0.0, "vmax": error_max}))
    if x_warped is not None:
        rows.append(("motion prediction", lambda p: _mid_slice(x_warped, p), gray))
    rows.append(("velocity", lambda p: velocity_rgb(v, p, v_scale), {}))

    planes = list(PLANES)
    # The last, narrow column holds the colorbars of the gray images (in the first row) and of the error.
    fig, axes = plt.subplots(len(rows), len(planes) + 1, figsize=(3.5 * len(planes) + 0.6, 2.6 * len(rows)), squeeze=False,
                             width_ratios=[1.0] * len(planes) + [0.04], layout="constrained")
    for r, (row_name, image, kwargs) in enumerate(rows):
        for c, plane in enumerate(planes):
            im = _imshow(axes[r, c], image(plane), plane, extent, **kwargs)
            if r == 0:
                axes[r, c].set_title(plane, fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(row_name, fontsize=9)
        if r == 0 or row_name.startswith("|"):
            fig.colorbar(im, cax=axes[r, -1], label="mm$^{-1}$")
        else:
            axes[r, -1].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.supxlabel(f"velocity: R, G, B = x (W), y (H), z (D) component, gray = 0, saturated at ±{v_scale:.3g} mm", fontsize=8)
    _save_atomic(fig, path)


class _Quiver:
    """Arrows of the in-plane velocity of a mid plane on a grid with the same physical spacing in both directions, n_arrows along the
    longest side of the planes. An arrow is gain times as long as the velocity (both in mm), and colored by the in-plane speed."""

    def __init__(self, shape: tuple[int, int, int], extent: tuple[float, float, float], plane: str, n_arrows: int):
        _, (self.row_axis, self.col_axis) = PLANES[plane]
        self.plane = plane
        spacing = max(extent) / n_arrows
        self.index = []
        self.coords = []
        for axis in (self.row_axis, self.col_axis):
            pixel = extent[axis] / shape[axis]
            step = max(1, round(spacing / pixel))
            index = np.arange(step // 2, shape[axis], step)
            self.index.append(index)
            self.coords.append((index + 0.5) * pixel)

    def uv(self, vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The (col, row) components of the velocity at the arrows."""
        sl = _mid_slice(vt, self.plane)[np.ix_(*self.index)]
        return sl[..., 2 - self.col_axis], sl[..., 2 - self.row_axis]


def in_plane_speeds(v: list[np.ndarray] | np.ndarray, shape: tuple[int, int, int], extent: tuple[float, float, float], n_arrows: int) -> np.ndarray:
    """The in-plane speeds at the arrows of all planes and the given velocities."""
    speeds = []
    for plane in PLANES:
        q = _Quiver(shape, extent, plane, n_arrows)
        for vt in np.reshape(v, (-1, *shape, 3)):
            u, w = q.uv(vt)
            speeds.append(np.hypot(u, w).ravel())
    return np.concatenate(speeds)


def quiver_scales(speeds: np.ndarray, extent: tuple[float, float, float], n_arrows: int) -> tuple[float, float]:
    """The arrow gain at which the 90th percentile of the in-plane speeds (see in_plane_speeds, e.g. of several studies to compare them)
    is drawn as long as the arrow spacing, and the 99th percentile as the top of the color scale (the speeds have long tails, so that the
    maximum would leave most arrows unreadable). For studies of different extents, pass the largest."""
    p90, p99 = np.percentile(speeds, [90, 99])
    return (max(extent) / n_arrows / p90 if p90 > 0 else 1.0), (float(p99) or 1.0)


def _quiver_figure(background: np.ndarray, vt: np.ndarray, extent: tuple[float, float, float], n_arrows: int, gain: float,
                   vmin: float, vmax: float, speed_max: float):
    planes = list(PLANES)
    fig, axes = plt.subplots(1, len(planes), figsize=(4.5 * len(planes), 4.6), dpi=100)
    # NOTE: a fixed layout (and a whole number of pixels), as in plot_prediction_gif, so that PillowWriter renders the frames correctly.
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.08, top=0.88, wspace=0.05)
    artists = []
    for ax, plane in zip(axes, planes):
        q = _Quiver(background.shape, extent, plane, n_arrows)
        im = _imshow(ax, _mid_slice(background, plane), plane, extent, cmap="gray", vmin=vmin, vmax=vmax)
        u, w = q.uv(vt)
        X, Y = np.meshgrid(q.coords[1], q.coords[0])
        arrows = ax.quiver(X, Y, u, w, np.hypot(u, w), cmap="plasma", norm=plt.Normalize(0.0, speed_max, clip=True), angles="xy", scale_units="xy",
                           scale=1.0 / gain, width=0.004, headwidth=3.5, headlength=4)
        ax.set_title(plane, fontsize=10)
        artists.append((im, arrows, q, plane))
    cax = fig.add_axes((0.935, 0.12, 0.012, 0.72))
    fig.colorbar(artists[0][1], cax=cax, label="in-plane speed (mm)", extend="max")
    fig.text(0.47, 0.025, f"arrows {gain:.3g} x the velocity, on the reconstruction", ha="center", fontsize=8)
    return fig, artists


def plot_velocity_quiver(
    background: np.ndarray,
    vt: np.ndarray,
    extent: tuple[float, float, float],
    path: Path,
    title: str,
    n_arrows: int = 24,
    gain: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    speed_max: float | None = None,
) -> None:
    """The in-plane velocity vt of one time bin as arrows on top of background (e.g. the reconstruction) in the three mid planes. Pass
    gain and speed_max (see quiver_scales) and vmin/vmax of the whole study (or of all studies) to compare time bins."""
    if gain is None or speed_max is None:
        default_gain, default_speed_max = quiver_scales(in_plane_speeds(vt, background.shape, extent, n_arrows), extent, n_arrows)
        gain, speed_max = gain or default_gain, speed_max or default_speed_max
    vmin = float(background.min()) if vmin is None else vmin
    vmax = float(background.max()) if vmax is None else vmax
    fig, _ = _quiver_figure(background, vt, extent, n_arrows, gain, vmin, vmax, speed_max)
    fig.suptitle(title, fontsize=11)
    _save_atomic(fig, path)


def plot_velocity_quiver_gif(
    background: list[np.ndarray],
    v: list[np.ndarray],
    extent: tuple[float, float, float],
    path: Path,
    title: str,
    n_arrows: int = 24,
    fps: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    gain: float | None = None,
    speed_max: float | None = None,
) -> None:
    """plot_velocity_quiver over the time bins, with background[t] (e.g. the reconstructions) and v[t], and common scales (by default
    those of all the time bins)."""
    if gain is None or speed_max is None:
        default_gain, default_speed_max = quiver_scales(in_plane_speeds(np.stack(v), background[0].shape, extent, n_arrows), extent, n_arrows)
        gain, speed_max = gain or default_gain, speed_max or default_speed_max
    vmin = float(min(b.min() for b in background)) if vmin is None else vmin
    vmax = float(max(b.max() for b in background)) if vmax is None else vmax
    fig, artists = _quiver_figure(background[0], v[0], extent, n_arrows, gain, vmin, vmax, speed_max)
    suptitle = fig.suptitle("", fontsize=11)

    def update(t: int):
        for im, arrows, q, plane in artists:
            im.set_data(_mid_slice(background[t], plane))
            u, w = q.uv(v[t])
            arrows.set_UVC(u, w, np.hypot(u, w))
        suptitle.set_text(f"{title}, time bin {t + 1}/{len(v)}")
        return [a for im, arrows, _, _ in artists for a in (im, arrows)] + [suptitle]

    update(0)
    _save_gif(fig, update, len(v), path, fps)


def _save_gif(fig, update, frames: int, path: Path, fps: int) -> None:
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    anim.save(tmp, writer=PillowWriter(fps=fps))
    plt.close(fig)
    tmp.replace(path)


def _sinogram_slices(y: np.ndarray, angles: tuple[float, float], detector_extent: tuple[float, float]) -> list[tuple[str, np.ndarray, tuple]]:
    """The (title, image, extent) of the mid slices of a sinogram (views, cols, rows): the middle projection (rows x cols), the
    sinogram of the middle detector row (views x cols) and that of the middle detector column (views x rows)."""
    n_views, n_cols, n_rows = y.shape
    a0, a1 = np.degrees(angles)
    view = n_views // 2
    c, r = detector_extent[0] / 2, detector_extent[1] / 2
    return [
        (f"projection at {a0 + (view + 0.5) * (a1 - a0) / n_views:.0f}°", y[view].T, (-c, c, -r, r)),
        ("sinogram, middle detector row", y[:, :, n_rows // 2], (-c, c, a0, a1)),
        ("sinogram, middle detector column", y[:, n_cols // 2, :], (-r, r, a0, a1)),
    ]


SINOGRAM_LABELS = [("detector column (mm)", "detector row (mm)"), ("detector column (mm)", "angle (°)"), ("detector row (mm)", "angle (°)")]


def sinogram_scales(g: list[np.ndarray], reprojection: list[np.ndarray] | None) -> tuple[float, float, float]:
    """(vmin, vmax) of the data and the 99.9th percentile of the absolute residual over the time bins, for plot_sinogram(_gif)."""
    vmin, vmax = float(min(gt.min() for gt in g)), float(max(gt.max() for gt in g))
    residual_max = float(np.percentile(np.abs(np.stack(reprojection) - np.stack(g)), 99.9)) if reprojection is not None else 1.0
    return vmin, vmax, residual_max or 1.0


def _sinogram_figure(g: list[np.ndarray], reprojection: list[np.ndarray] | None, angles: list[tuple[float, float]],
                     detector_extent: tuple[float, float], vmin: float, vmax: float, residual_max: float):
    """The figure of the sinogram slices of time bin 0, and update(t) to show time bin t."""
    gray = {"cmap": "gray", "vmin": vmin, "vmax": vmax}
    rows = [("data", lambda t: g[t], gray)]
    if reprojection is not None:
        rows.append(("reprojection", lambda t: reprojection[t], gray))
        rows.append(("reprojection - data", lambda t: reprojection[t] - g[t], {"cmap": "RdBu_r", "vmin": -residual_max, "vmax": residual_max}))
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.2 * len(rows)), dpi=100, squeeze=False)
    # NOTE: a fixed layout (and a whole number of pixels), as in plot_prediction_gif, so that PillowWriter renders the frames correctly.
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.08, top=0.9, wspace=0.25, hspace=0.45)
    artists = []
    for r, (row_name, sinogram, kwargs) in enumerate(rows):
        for c, (_, image, extent) in enumerate(_sinogram_slices(sinogram(0), angles[0], detector_extent)):
            ax = axes[r, c]
            im = ax.imshow(image, origin="lower", extent=extent, aspect="auto", interpolation="nearest", **kwargs)
            ax.set_xlabel(SINOGRAM_LABELS[c][0], fontsize=8)
            ax.set_ylabel(SINOGRAM_LABELS[c][1], fontsize=8)
            ax.tick_params(labelsize=7)
            artists.append((im, ax.set_title("", fontsize=9), sinogram, row_name, c))
        bbox = axes[r, -1].get_position()
        fig.colorbar(im, cax=fig.add_axes((0.915, bbox.y0, 0.01, bbox.height)), label="line integral" if r < 2 else "difference")
        axes[r, 0].annotate(row_name, xy=(-0.3, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center", fontsize=10)

    def update(t: int) -> list:
        for im, title, sinogram, row_name, c in artists:
            slice_title, image, extent = _sinogram_slices(sinogram(t), angles[t], detector_extent)[c]
            im.set_data(image)
            im.set_extent(extent)
            title.set_text(slice_title)
        return [a for im, title, *_ in artists for a in (im, title)]

    update(0)
    return fig, update


def plot_sinogram(
    g: list[np.ndarray],
    angles: list[tuple[float, float]],
    detector_extent: tuple[float, float],
    t: int,
    path: Path,
    title: str,
    reprojection: list[np.ndarray] | None = None,
    scales: tuple[float, float, float] | None = None,
) -> None:
    """The mid slices of the data g[t] of time bin t and (if given) of the reprojection of the reconstruction and its difference to the
    data. scales is (vmin, vmax, residual_max), see sinogram_scales, by default of all the time bins."""
    fig, update = _sinogram_figure(g, reprojection, angles, detector_extent, *(scales or sinogram_scales(g, reprojection)))
    update(t)
    fig.suptitle(title, fontsize=11)
    _save_atomic(fig, path)


def plot_sinogram_gif(
    g: list[np.ndarray],
    angles: list[tuple[float, float]],
    detector_extent: tuple[float, float],
    path: Path,
    title: str,
    reprojection: list[np.ndarray] | None = None,
    scales: tuple[float, float, float] | None = None,
    fps: int = 3,
) -> None:
    """plot_sinogram over the time bins."""
    fig, update_bin = _sinogram_figure(g, reprojection, angles, detector_extent, *(scales or sinogram_scales(g, reprojection)))
    suptitle = fig.suptitle("", fontsize=11)

    def update(t: int) -> list:
        suptitle.set_text(f"{title}, time bin {t + 1}/{len(g)}")
        return update_bin(t) + [suptitle]

    update(0)
    _save_gif(fig, update, len(g), path, fps)
