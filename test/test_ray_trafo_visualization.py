"""
Graphical inspection of RayTransform's 3D geometries on real patient data: forward-projects one
respiratory-phase volume through each of five geometry configurations and saves a gif that sweeps
through the detector's row axis (per ray_trafo.py's (col, row) detector-axis convention -- see the
NOTE in RayTransform.__init__ -- row is the axis physically aligned with `rotAxis`), one frame per
row, each frame showing the full (view, col) projection image at that row.

All five cases share the same nDetectorCols/nDetectorRows/extents/GantrySpeed/nViews/Flux/rotAxis
(taken directly from ray_trafo.DEFAULT_CONFIG plus rotAxis, which DEFAULT_CONFIG doesn't include since
it's only meaningful once you know the reco_space's axis order) so the outputs are visually
comparable -- only the geometry-defining parameters actually differ per case:

  (A) Default config from ray_trafo.py (ConeBeamGeometry, no pitch).
  (B) Helical cone-beam: same as (A) but pitch=50.
  (C) Parallel3dAxisGeometry (no source/detector radius, no curvature, no pitch -- parallel geometries
      don't take them).
  (D) Same as (C), through RayTransform.simulate_noise() instead of the clean forward projection.
  (E) Cone-beam nominally with a spherical detector (see FORCED-FLAT note below).

FORCED-FLAT NOTE: (A)/(B) nominally use DEFAULT_CONFIG's cylindrical detector
(`det_curvature_radius=(950, inf)`) and (E) nominally uses a spherical one (`(950, 950)`), but no
projector backend in this ODL install can actually render either. Confirmed exhaustively, not just for
the default `impl`: `odl.applications.tomo.backends` contains exactly four modules (astra_cpu,
astra_cuda, astra_setup, skimage_radon); astra_setup.astra_projection_geometry() is the single dispatch
function both astra_cpu and astra_cuda call, and every one of its six branches (2D/3D x
parallel/divergent x cpu/cuda) requires `isinstance(geometry.detector, (Flat1dDetector,
Flat2dDetector))`, with no curved-detector branch anywhere; skimage_radon is hardcoded to
`Parallel2dGeometry` only. So for every case that would otherwise use a curved detector, this script
forces `det_curvature_radius=None` (flat) instead, purely so a gif can be produced at all, and says so
in that case's title/log line. Under this forcing, (A) and (E) end up as the literal same geometry
(same radii, pitch=0, flat detector) -- the log makes that explicit rather than silently producing two
identical-looking gifs.

This is a visualization/smoke-test script, not a pass/fail correctness suite (see test_ray_trafo.py for
that) -- it reports shape/finiteness per case and where each gif was written, so problems are visible
either numerically or by opening the gif.

Run directly: python test/test_ray_trafo_visualization.py
"""
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from operators.tomo.ray_trafo import RayTransform, DEFAULT_CONFIG

PLOT_DIR = Path("test/plots/ray_trafo")
REAL_DATA_ROOT = Path("/media/trulssv/LDDMM")

# Parameters shared by every case in this file, taken directly from DEFAULT_CONFIG so all five
# outputs are comparable. rotAxis is added on top since DEFAULT_CONFIG (in ray_trafo.py) doesn't
# include it -- it's a property of how the reco_space's axes are used, not of the detector.
SHARED = {
    "nDetectorCols": DEFAULT_CONFIG["nDetectorCols"],
    "nDetectorRows": DEFAULT_CONFIG["nDetectorRows"],
    "DetectorColExtent": DEFAULT_CONFIG["DetectorColExtent"],
    "DetectorRowExtent": DEFAULT_CONFIG["DetectorRowExtent"],
    "GantrySpeed": DEFAULT_CONFIG["GantrySpeed"],
    "nViews": DEFAULT_CONFIG["nViews"],
    "Flux": DEFAULT_CONFIG["Flux"],
    "rotAxis": (1.0, 0.0, 0.0),
}


def load_real_patient_frame(quality="high", mode="test"):
    """Loads one respiratory-phase frame + the physical (D,H,W) extent computed from its meta_data.
    Downsamples the reco grid so a full DEFAULT_CONFIG-scale (512x96 detector, 512 views) forward
    projection stays fast -- this is about visual inspection of the geometry, not resolution.
    """
    if not REAL_DATA_ROOT.exists():
        return None
    from data.data_loaders import RegAndReconDataset

    dataset = RegAndReconDataset(qualities=[quality], mode=mode, data_root=REAL_DATA_ROOT)
    if len(dataset) == 0:
        return None
    sample = dataset[0]
    volume = sample["volume_processed"].float()  # (T, D, H, W)
    meta_data = sample["meta_data"]
    frame = volume[0]  # (D, H, W), first respiratory phase

    d, h, w = frame.shape
    step_h, step_w = max(h // 96, 1), max(w // 96, 1)
    frame_small = frame[:, ::step_h, ::step_w]

    pixel_spacing = meta_data["resampled_pixel_spacing"]
    slice_thickness = meta_data["resampled_slice_thickness"]
    extent = (d * slice_thickness, h * pixel_spacing[1], w * pixel_spacing[0])  # unchanged by subsampling
    return frame_small, tuple(extent)


def plot_row_sweep_gif(sino: torch.Tensor, savepath: Path, title: str, fps: int = 8, cmap: str = "gray") -> Path:
    """sino: (views, cols, rows) tensor (single batch element already squeezed out). Animates one
    frame per detector row index, each frame the full (view, col) projection image at that row --
    i.e. sweeping through the detector's row axis.
    """
    import textwrap
    title = "\n".join(textwrap.fill(line, width=55) for line in title.split("\n"))

    arr = sino.detach().cpu().numpy()
    n_rows = arr.shape[-1]
    frames = [arr[..., r] for r in range(n_rows)]
    vmin, vmax = float(arr.min()), float(arr.max())

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ttl = ax.set_title(f"{title}\n(detector row 1/{n_rows})", fontsize=9)
    ax.set_xlabel("detector col")
    ax.set_ylabel("view")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    def update(i):
        im.set_data(frames[i])
        ttl.set_text(f"{title}\n(detector row {i + 1}/{n_rows})")
        return [im, ttl]

    anim = FuncAnimation(fig, update, frames=n_rows, interval=1000 / fps, blit=False)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    anim.save(savepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return savepath


def build_cases():
    """Returns [(name, build_rt_fn, apply_fn)]. apply_fn(rt, x) -> projection tensor to visualize --
    identical to rt(x) for every case except (D), which routes through simulate_noise instead.
    """
    def clean_forward(rt, x):
        return rt(x)

    def noisy_forward(rt, x):
        # simulate_noise expects roughly-HU-scaled input (see its docstring: (x/1000+1)*mu_water); the
        # real volume here is a normalized attenuation-like map in ~[0,4], not HU, so it's rescaled to
        # a plausible soft-tissue HU range purely for this demo's noise visualization.
        x_hu = (x - x.mean()) * 500.0
        return rt.simulate_noise(x_hu)

    # det_curvature_radius is forced to None (flat) for every case below that would otherwise want
    # curvature -- see the FORCED-FLAT NOTE in the module docstring for why (no available backend can
    # render a curved 3D detector). DEFAULT_CONFIG's actual (950, inf) and (E)'s intended (950, 950)
    # are recorded in each case's `note` purely for the log/title, not passed to RayTransform.
    cases = [
        ("A_default_conebeam", lambda: RayTransform(
            geometry="ConeBeamGeometry", **SHARED,
            source_radius=DEFAULT_CONFIG["source_radius"], det_radius=DEFAULT_CONFIG["det_radius"],
            det_curvature_radius=None, pitch=DEFAULT_CONFIG["pitch"],
            extent=None, shape=None,
        ), clean_forward,
         "FORCED FLAT: DEFAULT_CONFIG wants det_curvature_radius=(950, inf) (cylindrical); no backend supports curved detectors"),

        ("B_helical_conebeam_pitch50", lambda: RayTransform(
            geometry="ConeBeamGeometry", **SHARED,
            source_radius=DEFAULT_CONFIG["source_radius"], det_radius=DEFAULT_CONFIG["det_radius"],
            det_curvature_radius=None, pitch=50.0,
            extent=None, shape=None,
        ), clean_forward,
         "FORCED FLAT: DEFAULT_CONFIG wants det_curvature_radius=(950, inf) (cylindrical); no backend supports curved detectors"),

        ("C_parallel3d", lambda: RayTransform(
            geometry="Parallel3dAxisGeometry", **SHARED,
            source_radius=None, det_radius=None,
            extent=None, shape=None,
        ), clean_forward, ""),

        ("D_parallel3d_noisy", lambda: RayTransform(
            geometry="Parallel3dAxisGeometry", **SHARED,
            source_radius=None, det_radius=None,
            extent=None, shape=None,
        ), noisy_forward, ""),

        # NOTE on det_curvature_radius: it's (first, second) matching dpart's (col, row) axis order
        # (see the NOTE in RayTransform.__init__), not literally "row radius, col radius". (950, inf)
        # would mean "curved along the first detector axis, flat along the second" -- a cylindrical
        # detector; (950, 950) (equal radii) would curve identically in both directions -- spherical.
        # Neither is renderable here (see FORCED-FLAT NOTE), so this ends up identical to (A): same
        # radii, pitch=0, flat detector.
        ("E_spherical_detector_conebeam", lambda: RayTransform(
            geometry="ConeBeamGeometry", **SHARED,
            source_radius=DEFAULT_CONFIG["source_radius"], det_radius=DEFAULT_CONFIG["det_radius"],
            det_curvature_radius=None, pitch=0.0,
            extent=None, shape=None,
        ), clean_forward,
         "FORCED FLAT: intended det_curvature_radius=(950, 950) (spherical); no backend supports curved detectors -- geometrically identical to (A) once flattened"),
    ]
    return cases


def main():
    real = load_real_patient_frame()
    if real is None:
        print(f"SKIPPED: {REAL_DATA_ROOT} not available in this environment -- nothing to visualize.")
        return 1

    frame, extent = real
    reco_shape = tuple(frame.shape)
    x = frame.unsqueeze(0)
    print(f"Loaded real patient frame, shape={reco_shape}, physical extent (D,H,W)={extent} mm")

    results = []
    for name, build_rt, apply_fn, note in build_cases():
        try:
            if not name=="D_parallel3d_noisy":
                assert False, f"Skipping test for {name}"
            rt = build_rt()
            rt._init_ray_transform(shape=reco_shape, extent=extent, time_steps=1.0)
            proj = apply_fn(rt, x)
            finite = torch.isfinite(proj).all().item()

            title = name.replace("_", " ")
            if note:
                title += f"\n[{note}]"
            savepath = PLOT_DIR / f"{name}.gif"
            plot_row_sweep_gif(proj[0], savepath, title=title)

            results.append({"name": name, "status": "OK" if finite else "NON-FINITE",
                             "shape": tuple(proj.shape), "path": savepath, "note": note})
        except Exception as exc:
            results.append({"name": name, "status": "ERROR", "detail": f"{type(exc).__name__}: {exc}", "note": note})

    name_w = max(len(r["name"]) for r in results)
    print(f"\n{'case':{name_w}}  status      shape                  gif")
    for r in results:
        if r["status"] == "ERROR":
            print(f"{r['name']:{name_w}}  {r['status']:10}  {r['detail']}")
        else:
            print(f"{r['name']:{name_w}}  {r['status']:10}  {str(r['shape']):20}  {r['path']}")
        if r.get("note"):
            print(f"{'':{name_w}}  {'':10}  note: {r['note']}")

    n_bad = sum(1 for r in results if r["status"] != "OK")
    return 1 if n_bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
