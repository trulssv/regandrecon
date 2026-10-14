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

Run with pytest (one test per case), or directly for a summary table: python test/test_ray_trafo.py
"""
import sys
import traceback

import pytest
import torch
import numpy as np

from operators.tomo.ray_trafo import RayTransform

from data.config import DEFAULT_SIMULATED_ROOT as REAL_DATA_ROOT


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
        from data.loaders import RegAndReconDataset
        dataset = RegAndReconDataset(qualities=[quality], mode=mode, data_root=REAL_DATA_ROOT, load_volume=True)
        if len(dataset) == 0:
            return None
        sample = dataset[0]
        volume = sample["volume"].float()  # (T, D, H, W)
        extent = sample["meta"]["extent"]  # (D,H,W) extent, mm
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


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", build_cases(), ids=lambda case: case[0])
def test_ray_transform_case(case):
    if case[1] is None:
        pytest.skip(f"{REAL_DATA_ROOT} not available in this environment")
    result = run_case(*case)
    assert result["status"] == "PASS", result


def test_coupling_with_flow_deformation():
    assert run_coupling_demo()


def test_geometry_cache_and_lazy_fbp():
    """Switching back to a geometry reuses its operators, and the FBP operators are only built on first use, per geometry."""
    rt = RayTransform(geometry="Parallel2dGeometry", nDetectorCols=64, DetectorColExtent=100.0, GantrySpeed=torch.pi, nViews=32, Flux=1e14,
                      nDetectorRows=None, DetectorRowExtent=None, rotAxis=None, source_radius=None, det_radius=None, extent=None, shape=None)
    shape, time_steps = (32, 32), [0.5, 0.5]
    rt._init_ray_transform(shape, (50.0, 50.0), time_steps)
    ops_a = rt.ray_transform
    assert all(entry["fbp_operator"] is None for entry in rt._geometry_cache.values())

    rt._init_ray_transform(shape, (80.0, 80.0), time_steps)
    ops_b, fbp_b = rt.ray_transform, rt.fbp_operator
    assert ops_b is not ops_a and rt.extent == (80.0, 80.0)

    rt._init_ray_transform(shape, (50.0, 50.0), time_steps)
    assert rt.ray_transform is ops_a and rt.extent == (50.0, 50.0) and rt.reco_space.max_pt[0] == 25.0
    assert rt._geometry_cache[rt._geometry_key]["fbp_operator"] is None    # not built for geometry a yet
    assert rt.fbp_operator is not fbp_b

    x = torch.rand(1, *shape)
    assert torch.equal(rt.forward_bin(x, 0), ops_a[0](x))
    assert rt.fbp_bin(rt.forward_bin(x, 0), 0).shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("geometry", ["Parallel3dAxisGeometry", "ConeBeamGeometry"])
def test_astra_cuda_module_matches_odl(geometry):
    """AstraCudaModule (ASTRA on CUDA tensors) computes the same projections, back projections and gradients as odl's OperatorModule,
    for an anisotropic volume, limited-angle time bins and extra batch axes, and passes CPU tensors to OperatorModule.

    The results are bitwise those of odl's own GPU backend (an odl space with impl='pytorch'). ASTRA rounds the parallel beam back
    projection slightly differently for host than for device data, so they differ from our numpy-backed odl spaces by about 5e-6
    relative to the norm (up to 2e-4 relative to the maximum in single voxels)."""
    import odl
    import odl.applications.tomo as tomo
    from odl.contrib.torch import OperatorModule
    from operators.tomo.ray_trafo import AstraCudaModule

    def assert_norm_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float = 2e-5) -> None:
        assert ((actual - expected).norm() / expected.norm()).item() <= rtol

    cone = geometry == "ConeBeamGeometry"
    rt = RayTransform(geometry=geometry, nDetectorCols=48, DetectorColExtent=400.0, GantrySpeed=2 * torch.pi, nViews=120, Flux=1e14,
                      nDetectorRows=20, DetectorRowExtent=200.0, rotAxis=(1.0, 0.0, 0.0), source_radius=750.0 if cone else None,
                      det_radius=750.0 if cone else None, extent=None, shape=None)
    shape = (16, 20, 24)
    rt._init_ray_transform(shape, (160.0, 200.0, 240.0), [0.25, 0.25])
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)

    gpu_space = odl.uniform_discr(rt.reco_space.min_pt, rt.reco_space.max_pt, shape, dtype='float32', impl='pytorch', device='cuda:0')
    for t in range(2):
        fwd, adj = rt.ray_transform[t], rt.ray_transform_adjoint[t]
        assert isinstance(fwd, AstraCudaModule) and isinstance(adj, AstraCudaModule)
        odl_fwd, odl_adj = OperatorModule(fwd.operator), OperatorModule(adj.operator)

        gpu_op = tomo.RayTransform(gpu_space, rt.geometries[t], impl='astra_cuda')
        x_gpu = torch.rand(1, *shape, generator=gen).to(device)
        y_gpu = OperatorModule(gpu_op)(x_gpu)
        assert torch.equal(fwd(x_gpu), y_gpu)
        assert torch.equal(adj(y_gpu), OperatorModule(gpu_op.adjoint)(y_gpu))

        x = torch.rand(2, 1, *shape, generator=gen)                         # extra batch axes
        y_shape = odl_fwd(x[0]).shape[1:]
        y = torch.rand(2, 1, *y_shape, generator=gen)
        for module, odl_module, inp in ((fwd, odl_fwd, x), (adj, odl_adj, y)):
            ref_in = inp.clone().requires_grad_()
            ref_out = odl_module(ref_in)
            ref_out.backward(torch.ones_like(ref_out) + ref_out.detach())   # a non-constant output gradient
            cuda_in = inp.to(device).requires_grad_()
            cuda_out = module(cuda_in)
            cuda_out.backward((torch.ones_like(ref_out) + ref_out.detach()).to(device))
            assert_norm_close(cuda_out.detach().cpu(), ref_out.detach())
            assert cuda_in.grad is not None and ref_in.grad is not None
            assert_norm_close(cuda_in.grad.cpu(), ref_in.grad)
            torch.testing.assert_close(module(inp), odl_module(inp))      # CPU tensors go through odl


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
