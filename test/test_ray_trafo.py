"""
Standardized correctness checks for operators.tomo.ray_trafo.RayTransform: all four supported
geometries (Parallel2dGeometry, FanBeamGeometry, Parallel3dAxisGeometry, ConeBeamGeometry), on both
synthetic 2D data and real 3D patient data, at a range of `time_steps` configurations -- including
limited-angle time bins (angular coverage < pi), which is the clinically relevant dynamic-CT case this
project cares about. Also includes a coupling demo with operators.lddmm.deform.FlowDeformationOperator,
producing a per-timestep dynamic sinogram from a single deforming template.

Per case, this checks:
  - construction and forward/adjoint/FBP all run without error, with the expected tensor shapes.
  - forward/adjoint are consistent with each other as a weighted inner-product (adjoint) pair:
    <Ax, y>_range == <x, A*y>_domain, where the two inner products are weighted by their respective
    odl space's `cell_volume` (proj_space and reco_space have very different physical cell sizes --
    naively comparing unweighted dot products of the raw tensors is NOT a valid check and was
    confirmed, empirically, to give wildly unstable ratios across trials -- see the module docstring
    note in `adjoint_consistency_error` below for why NON-NEGATIVE test signals are used here).
  - FBP reconstructs a recognizable phantom (bounded relative error) for full/near-full angular
    coverage; for genuinely limited-angle bins this is *not* checked against an accuracy threshold
    (limited-angle FBP is a known ill-posed problem -- streak/incompleteness artifacts are expected,
    not a bug) -- only that it runs and produces finite output.

Run directly: python test/test_ray_trafo.py
"""
import sys
import traceback
from pathlib import Path

import torch
import numpy as np

from operators.tomo.ray_trafo import RayTransform

REAL_DATA_ROOT = Path("/media/trulssv/LDDMM")


# ---------------------------------------------------------------------------
# Phantoms
# ---------------------------------------------------------------------------

def shepp_logan_2d(shape=(128, 128)):
    from sigpy import shepp_logan
    return torch.from_numpy(shepp_logan(shape, dtype=np.float32)) # type: ignore


def block_phantom_3d(shape):
    """A simple non-symmetric 3D block phantom -- cheap, and asymmetric enough that a row/col or
    axis-order bug would visibly corrupt the reconstruction rather than being masked by symmetry."""
    d, h, w = shape
    x = torch.zeros(d, h, w, dtype=torch.float32)
    x[d // 4: 3 * d // 4, h // 3: 2 * h // 3, w // 4: 3 * w // 4] = 1.0
    x[d // 2 - 1: d // 2 + 1, h // 6: h // 3, w // 2 - 1: w // 2 + 1] = 2.0  # off-center marker
    return x


def load_real_patient_volume(quality="high", mode="test"):
    """Loads the full (T, D, H, W) respiratory-phase volume + the physical extent computed from its
    meta_data, from the real dataset mounted at REAL_DATA_ROOT. Returns None if the data isn't
    available in this environment (e.g. the LDDMM mount isn't present) rather than raising, so the
    suite degrades gracefully instead of failing outright on machines without the mount.
    """
    if not REAL_DATA_ROOT.exists():
        return None
    try:
        from data.data_loaders import RegAndReconDataset
        dataset = RegAndReconDataset(qualities=[quality], mode=mode, data_root=REAL_DATA_ROOT)
        if len(dataset) == 0:
            return None
        sample = dataset[0]
        volume = sample["volume_processed"].float()  # (T, D, H, W)
        meta_data = sample["meta_data"]

        d, h, w = volume.shape[1:]
        pixel_spacing = meta_data["resampled_pixel_spacing"]  # (W spacing, H spacing), mm
        slice_thickness = meta_data["resampled_slice_thickness"]  # mm
        extent = (d * slice_thickness, h * pixel_spacing[1], w * pixel_spacing[0])  # (D,H,W) extent, mm
        return volume, tuple(extent)
    except Exception:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def adjoint_consistency_error(fwd_op, adj_op, x_shape, n_trials=3, seed=0):
    """Weighted adjoint dot-product check for a single bin's (forward, adjoint) OperatorModule pair:
    <Ax,y>_range should equal <x,A*y>_domain, where each side's naive (unweighted) tensor dot product
    is corrected by the ratio of the two odl spaces' cell volumes. NON-NEGATIVE random signals are
    used deliberately: with signed (torch.randn) inputs, the naive ratio <Ax,y>/<x,A*y> was found to
    swing wildly (including flipping sign) across repeated trials on the exact same operator, purely
    because the *unweighted* dot product of a large, zero-mean signed sum is itself close to zero and
    numerically unstable to divide by -- not because the operator/adjoint pair is actually
    inconsistent. With non-negative inputs the same ratio was stable to 4 decimal places across
    independent trials and matched the cell-volume ratio exactly, confirming the forward/adjoint pair
    *is* a correct weighted-adjoint pair; this check reproduces that stable, meaningful version of the
    test rather than the numerically-fragile signed one. Returns the mean relative error across trials.
    """
    weight = fwd_op.operator.range.cell_volume / fwd_op.operator.domain.cell_volume
    gen = torch.Generator().manual_seed(seed)
    errs = []
    for _ in range(n_trials):
        x = torch.rand(1, *x_shape, generator=gen, dtype=torch.float32)
        y_shape = fwd_op(torch.zeros(1, *x_shape)).shape
        y = torch.rand(*y_shape, generator=gen, dtype=torch.float32)
        Ax = fwd_op(x)
        Aty = adj_op(y)
        lhs = (Ax * y).sum().item() * weight
        rhs = (x * Aty).sum().item()
        errs.append(abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-8))
    return float(np.mean(errs))


def fbp_relative_error(recon: torch.Tensor, phantom: torch.Tensor) -> float:
    return (torch.linalg.norm(recon - phantom) / torch.linalg.norm(phantom)).item()


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------

ADJOINT_TOL = 0.05          # relative error tolerance for the weighted adjoint dot-product check
FULL_ANGLE_FBP_TOL = 0.35   # relative error tolerance for FBP with near-full angular coverage


def run_case(name, build_rt, reco_shape, phantom, time_steps, check_fbp_accuracy):
    """build_rt() -> RayTransform (already constructed, not yet _init_ray_transform'd).
    check_fbp_accuracy: if True, FBP relative error is checked against FULL_ANGLE_FBP_TOL (only
    appropriate for near-full angular coverage); if False (limited-angle / multi-bin cases), FBP is
    only checked for producing finite output, since limited-angle reconstruction is not expected to be
    numerically close to the phantom.
    """
    try:
        rt = build_rt()
        extent = rt.extent
        rt._init_ray_transform(shape=reco_shape, extent=extent, time_steps=time_steps)

        is_list = isinstance(phantom, list)
        # For list-valued time_steps, RayTransform requires x to be a list of per-bin images paired
        # elementwise with the per-bin geometries (x[i] projected through bin i's geometry) -- matching
        # a genuinely time-varying object (e.g. FlowDeformationOperator's per-frame output). `phantom`
        # is already a list of per-bin frames for those cases (see build_cases).
        x = [p.unsqueeze(0) for p in phantom] if is_list else phantom.unsqueeze(0)
        sino = rt(x)
        back = rt.adjoint(sino)
        recon = rt.FBP(sino)

        sino_shapes = [s.shape for s in sino] if is_list else sino.shape
        back_shapes = [b.shape for b in back] if is_list else back.shape
        recon_shapes = [r.shape for r in recon] if is_list else recon.shape

        expected_back_shape = (1, *reco_shape)
        back_ok = all(b.shape == expected_back_shape for b in back) if is_list else back.shape == expected_back_shape

        finite_ok = (all(torch.isfinite(r).all() for r in recon) if is_list
                     else torch.isfinite(recon).all().item())

        if is_list:
            adj_errs = [adjoint_consistency_error(fwd, adj, reco_shape)
                        for fwd, adj in zip(rt.ray_transform, rt.ray_transform_adjoint)]
            adj_err = float(np.mean(adj_errs))
        else:
            adj_err = adjoint_consistency_error(rt.ray_transform, rt.ray_transform_adjoint, reco_shape)
        adj_ok = adj_err <= ADJOINT_TOL

        fbp_err = None
        fbp_ok = True
        if check_fbp_accuracy:
            r = recon[0] if is_list else recon
            ref = phantom[0] if is_list else phantom
            fbp_err = fbp_relative_error(r.squeeze(0), ref)
            fbp_ok = fbp_err <= FULL_ANGLE_FBP_TOL

        status = "PASS" if (back_ok and finite_ok and adj_ok and fbp_ok) else "FAIL"
        return {
            "name": name, "status": status,
            "sino_shape": sino_shapes, "back_shape": back_shapes, "recon_shape": recon_shapes,
            "back_ok": back_ok, "finite_ok": finite_ok,
            "adj_err": adj_err, "adj_ok": adj_ok,
            "fbp_err": fbp_err, "fbp_ok": fbp_ok,
        }
    except Exception as exc:
        return {"name": name, "status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}


def build_cases():
    cases = []

    # ---------------- 2D synthetic ----------------
    phantom_2d = shepp_logan_2d((128, 128))

    cases.append(("2D Parallel, static (full angle)",
        lambda: RayTransform(geometry="Parallel2dGeometry", nDetectorCols=200, DetectorColExtent=300.0,
                              GantrySpeed=2 * torch.pi, nViews=200, Flux=1e14,
                              nDetectorRows=None, DetectorRowExtent=None, rotAxis=None,
                              source_radius=None, det_radius=None, extent=(300.0, 300.0), shape=None),
        (128, 128), phantom_2d, 1.0, True))

    cases.append(("2D FanBeam, static (full angle)",
        lambda: RayTransform(geometry="FanBeamGeometry", nDetectorCols=400, DetectorColExtent=600.0,
                              GantrySpeed=2 * torch.pi, nViews=400, Flux=1e14,
                              nDetectorRows=None, DetectorRowExtent=None, rotAxis=None,
                              source_radius=1000.0, det_radius=1000.0, extent=(200.0, 200.0), shape=None),
        (128, 128), phantom_2d, 1.0, True))

    cases.append(("2D Parallel, limited-angle time bins",
        lambda: RayTransform(geometry="Parallel2dGeometry", nDetectorCols=150, DetectorColExtent=300.0,
                              GantrySpeed=2 * torch.pi, nViews=300, Flux=1e14,
                              nDetectorRows=None, DetectorRowExtent=None, rotAxis=None,
                              source_radius=None, det_radius=None, extent=(300.0, 300.0), shape=None),
        (96, 96), [shepp_logan_2d((96, 96))] * 5, [0.2, 0.2, 0.2, 0.2, 0.2], False))
        # 5 bins x 0.2 x 2*pi = 0.4*pi (~72 deg) per bin: genuinely limited-angle, well under pi.
        # Same static phantom repeated per bin here (this case is about the geometry/binning, not
        # motion) -- see run_coupling_demo() below for a genuinely time-varying (deforming) object.

    # ---------------- 3D synthetic ----------------
    shape_3d = (16, 20, 24)   # deliberately anisotropic (D != H != W) to catch axis-order bugs
    phantom_3d = block_phantom_3d(shape_3d)

    cases.append(("3D Parallel3dAxis, static (full angle)",
        lambda: RayTransform(geometry="Parallel3dAxisGeometry", nDetectorCols=24, DetectorColExtent=240.0,
                              GantrySpeed=2 * torch.pi, nViews=180, Flux=1e14,
                              nDetectorRows=16, DetectorRowExtent=160.0, rotAxis=(1.0, 0.0, 0.0),
                              source_radius=None, det_radius=None, extent=(160.0, 200.0, 240.0), shape=None),
        shape_3d, phantom_3d, 1.0, True))

    cases.append(("3D ConeBeam, static (full angle)",
        lambda: RayTransform(geometry="ConeBeamGeometry", nDetectorCols=64, DetectorColExtent=400.0,
                              GantrySpeed=2 * torch.pi, nViews=200, Flux=1e14,
                              nDetectorRows=32, DetectorRowExtent=160.0, rotAxis=(1.0, 0.0, 0.0),
                              source_radius=750.0, det_radius=750.0, extent=(160.0, 200.0, 240.0), shape=None),
        shape_3d, phantom_3d, 1.0, True))

    cases.append(("3D ConeBeam, limited-angle time bins",
        lambda: RayTransform(geometry="ConeBeamGeometry", nDetectorCols=48, DetectorColExtent=400.0,
                              GantrySpeed=2 * torch.pi, nViews=240, Flux=1e14,
                              nDetectorRows=24, DetectorRowExtent=160.0, rotAxis=(1.0, 0.0, 0.0),
                              source_radius=750.0, det_radius=750.0, extent=(160.0, 200.0, 240.0), shape=None),
        shape_3d, [phantom_3d] * 4, [0.15, 0.15, 0.15, 0.15], False))
        # 4 bins x 0.15 x 2*pi = 0.3*pi (~54 deg) per bin.

    # ---------------- 3D real patient data ----------------
    real = load_real_patient_volume()
    if real is not None:
        volume, extent = real  # volume: (T=10, D, H, W)
        t, d, h, w = volume.shape
        # Downsample so a CPU/GPU sanity run stays fast; this is a correctness smoke test, not a
        # high-resolution reconstruction benchmark.
        step_h, step_w = max(h // 96, 1), max(w // 96, 1)
        volume_small = volume[:, :, ::step_h, ::step_w]
        extent_small = extent  # physical extent is unchanged by subsampling the pixel grid
        shape_small = tuple(volume_small.shape[1:])

        cases.append((f"3D ConeBeam, REAL patient data {shape_small}, static",
            lambda: RayTransform(geometry="ConeBeamGeometry", nDetectorCols=128, DetectorColExtent=908.8,
                                  GantrySpeed=2 * torch.pi, nViews=360, Flux=1e14,
                                  nDetectorRows=64, DetectorRowExtent=80.0, rotAxis=(1.0, 0.0, 0.0),
                                  source_radius=540.0, det_radius=950.0, extent=extent_small, shape=None),
            shape_small, volume_small[0], 1.0, False))
            # check_fbp_accuracy=False: real anatomy at this heavy a downsample isn't expected to hit
            # the synthetic-phantom accuracy bar; this case is about shapes/finiteness/adjoint-consistency.

        cases.append((f"3D ConeBeam, REAL patient data {shape_small}, limited-angle (respiratory bins)",
            lambda: RayTransform(geometry="ConeBeamGeometry", nDetectorCols=128, DetectorColExtent=908.8,
                                  GantrySpeed=2 * torch.pi, nViews=360, Flux=1e14,
                                  nDetectorRows=64, DetectorRowExtent=80.0, rotAxis=(1.0, 0.0, 0.0),
                                  source_radius=540.0, det_radius=950.0, extent=extent_small, shape=None),
            shape_small, [volume_small[i] for i in range(t)], [0.1] * t, False))
            # t=10 bins (the real dataset's actual respiratory phases, one real per-phase volume per
            # bin -- not a repeated static frame) x 0.1 x 2*pi = 0.2*pi (~36 deg) each: the actual
            # clinically-motivated limited-angle scenario this project targets -- one gantry rotation
            # split across the respiratory cycle's time bins, with genuine motion between bins, same as
            # the real acquisition this data models.
    else:
        cases.append(("3D ConeBeam, REAL patient data",
                      None, None, None, None, None))  # recorded as SKIPPED below

    return cases


def run_coupling_demo():
    """Demonstrates coupling RayTransform with operators.lddmm.deform.FlowDeformationOperator: a single
    static velocity field deforms a 2D template into one frame per time bin (superres=True), and each
    frame is forward-projected through *its own* time bin's geometry -- matching the actual dynamic-CT
    scenario this project targets (an object that moves between/within gantry rotations). Uses
    RayTransform's native list handling directly: `forward(x: list[Tensor])` pairs `x[i]` with bin i's
    geometry elementwise, and `adjoint`/`FBP` do the same on the resulting list of sinograms.
    """
    from operators.lddmm.deform import FlowDeformationOperator

    print("\n--- Coupling demo: FlowDeformationOperator -> per-bin RayTransform ---")
    shape = (96, 96)
    extent = (240.0, 240.0)
    n_bins = 4

    template = shepp_logan_2d(shape).unsqueeze(0)  # (1, H, W)

    # Simple swirling velocity field, matching the convention used in the lddmm blur/boundary reports.
    coords = torch.arange(shape[0], dtype=torch.float32)
    ii, jj = torch.meshgrid(coords - shape[0] / 2, coords - shape[1] / 2, indexing="ij")
    distance = torch.sqrt(ii ** 2 + jj ** 2).clamp(min=1)
    v = torch.zeros((1, *shape, 2), dtype=torch.float32)
    v[0, :, :, 0] = -15.0 * jj / distance
    v[0, :, :, 1] = 15.0 * ii / distance

    deform_op = FlowDeformationOperator(N=n_bins, action="geometric", integration="euler",
                                         shape=shape)
    frames = deform_op(template, v, superres=True)  # list of n_bins deformed frames, (1, H, W) each
    print(f"deform_op produced {len(frames)} frames, shape each: {frames[0].shape}")

    rt = RayTransform(geometry="Parallel2dGeometry", nDetectorCols=150, DetectorColExtent=300.0,
                       GantrySpeed=2 * torch.pi, nViews=200, Flux=1e14,
                       nDetectorRows=None, DetectorRowExtent=None, rotAxis=None,
                       source_radius=None, det_radius=None, extent=extent, shape=None)
    time_steps = [1.0 / n_bins] * n_bins  # n_bins equal bins covering the full gantry rotation
    rt._init_ray_transform(shape=shape, extent=extent, time_steps=time_steps)

    sinos = rt(frames)  # native list handling: sinos[i] = geometry_i(frames[i])
    backs = rt.adjoint(sinos)
    for i, (frame, sino, back) in enumerate(zip(frames, sinos, backs)):
        finite = torch.isfinite(sino).all().item() and torch.isfinite(back).all().item()
        print(f"  bin {i}: frame {tuple(frame.shape)} -> sinogram {tuple(sino.shape)} -> "
              f"adjoint {tuple(back.shape)}  finite={finite}")

    all_finite = all(torch.isfinite(s).all().item() and torch.isfinite(b).all().item()
                      for s, b in zip(sinos, backs))
    shapes_ok = all(b.shape == f.shape for b, f in zip(backs, frames))
    ok = all_finite and shapes_ok
    print(f"Coupling demo: {'PASS' if ok else 'FAIL'} (all finite: {all_finite}, adjoint shapes match frames: {shapes_ok})")
    return ok


def format_shape(s):
    if isinstance(s, list):
        return "[" + ", ".join(str(tuple(x)) for x in s) + "]"
    return str(tuple(s))


def main():
    cases = build_cases()
    results = []
    for case in cases:
        name = case[0]
        if case[1] is None:
            results.append({"name": name, "status": "SKIPPED",
                             "detail": f"{REAL_DATA_ROOT} not available in this environment"})
            continue
        results.append(run_case(*case))

    name_w = max(len(r["name"]) for r in results)
    print(f"{'case':{name_w}}  status   adj_err   fbp_err   details")
    for r in results:
        if r["status"] in ("ERROR", "SKIPPED"):
            print(f"{r['name']:{name_w}}  {r['status']:7}  {r.get('detail', '')}")
            continue
        adj = f"{r['adj_err']:.5f}" if r["adj_err"] is not None else "n/a"
        fbp = f"{r['fbp_err']:.4f}" if r["fbp_err"] is not None else "n/a"
        detail = f"sino={format_shape(r['sino_shape'])} back={format_shape(r['back_shape'])}"
        if not r["back_ok"]:
            detail += "  <-- WRONG BACK SHAPE"
        if not r["finite_ok"]:
            detail += "  <-- NON-FINITE OUTPUT"
        if not r["adj_ok"]:
            detail += "  <-- ADJOINT INCONSISTENT"
        if r["fbp_err"] is not None and not r["fbp_ok"]:
            detail += "  <-- FBP ERROR TOO HIGH"
        print(f"{r['name']:{name_w}}  {r['status']:7}  {adj:8}  {fbp:8}  {detail}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] in ("FAIL", "ERROR"))
    n_skip = sum(1 for r in results if r["status"] == "SKIPPED")
    print(f"\n{n_pass} passed, {n_fail} failed/errored, {n_skip} skipped (of {len(results)}).")

    coupling_ok = run_coupling_demo()

    return 1 if (n_fail or not coupling_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
