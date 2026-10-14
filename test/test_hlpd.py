"""
Shape, gradient-flow and checkpointing checks for model.hlpd, run against a small mock ray transform so
that they don't depend on the ODL/ASTRA backend. The mock has the same interface as
operators.tomo.ray_trafo.RayTransform (list-valued __call__/adjoint and the per-time-bin forward_bin/adjoint_bin)
and is a random dense linear operator per time bin, mapping B x D x H x W volumes to B x nAngles x nCol x nRow
projections, with odl's (nAngles, nCol, nRow) projection axis order.
"""
import copy

import pytest
import torch

import model.hlpd as hlpd
from model.hlpd import (
    ConvStack,
    ConvStackParams,
    DeformParams,
    HLPDBlock,
    HLPDBlockParams,
    HLPDModel,
    HLPDUnrolled,
    HLPDUnrolledParams,
    NormalizationParams,
    RayTrafoParams,
)
from operators.lddmm.deform import FlowDeformationOperator
from operators.lddmm.helmholtz import HelmholtzOperator
from operators.lddmm.lddmm_loss import LDDMMloss
from operators.tomo.ray_trafo import operator_norm


T = 3
B = 2
SHAPE = (4, 6, 6)
RAY_PARAMS = RayTrafoParams(nDetectorCols=5, nDetectorRows=3, nViews=4)
SINO_SHAPE = (RAY_PARAMS.nViews, RAY_PARAMS.nDetectorCols, RAY_PARAMS.nDetectorRows)
CONV_PARAMS = ConvStackParams(hidden_channels=[8])


class MockRayTransform:
    """Random dense linear operator per time bin with the RayTransform interface."""

    def __init__(self, nDetectorCols: int, nDetectorRows: int, nViews: int, **kwargs) -> None:
        self.sino_shape = (nViews, nDetectorCols, nDetectorRows)

    def _init_ray_transform(self, shape: tuple[int, ...], extent: tuple[float, ...], time_steps: list[float]) -> None:
        self.shape = shape
        self.extent = extent
        generator = torch.Generator().manual_seed(0)
        n_dom = torch.Size(shape).numel()
        n_ran = torch.Size(self.sino_shape).numel()
        self.ray_transform = [torch.randn(n_ran, n_dom, generator=generator) / n_dom**0.5 for _ in time_steps]

    def forward_bin(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return (x.flatten(-3) @ self.ray_transform[t].T).unflatten(-1, self.sino_shape)

    def adjoint_bin(self, y: torch.Tensor, t: int) -> torch.Tensor:
        return (y.flatten(-3) @ self.ray_transform[t]).unflatten(-1, self.shape)

    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.forward_bin(xt, t) for t, xt in enumerate(x)]

    def adjoint(self, y: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.adjoint_bin(yt, t) for t, yt in enumerate(y)]

    def FBP(self, y: list[torch.Tensor]) -> list[torch.Tensor]:
        """A linear stand-in for the FBP, which is all that the scaling of the data needs."""
        return self.adjoint(y)


def make_ray_trafo() -> MockRayTransform:
    ray_trafo = MockRayTransform(nDetectorCols=RAY_PARAMS.nDetectorCols, nDetectorRows=RAY_PARAMS.nDetectorRows, nViews=RAY_PARAMS.nViews)
    ray_trafo._init_ray_transform(SHAPE, tuple(float(s) for s in SHAPE), [1.0] * T)
    return ray_trafo


def make_unrolled(recurrent: bool, shared_time: bool, shared_iters: bool, grad_checkpointing: bool, num_iters: int = 2) -> HLPDUnrolled:
    torch.manual_seed(0)
    p_in, p_out = (6, 1) if recurrent else (5 * T, T)
    d_in, d_out = (4, 1) if recurrent else (3 * T, T)
    m_in, m_out = (7, 3) if recurrent else (4 * T, 3 * T)
    block = HLPDBlock(
        ConvStack(p_in, p_out, **vars(CONV_PARAMS)),
        ConvStack(d_in, d_out, **vars(CONV_PARAMS)),
        ConvStack(m_in, m_out, **vars(CONV_PARAMS)),
        make_ray_trafo(),  # type: ignore[arg-type]
        recurrent=recurrent,
        time_steps=T,
        shared=shared_time,
    )
    return HLPDUnrolled(block, num_iters=num_iters, shared=shared_iters, grad_checkpointing=grad_checkpointing)


def make_inputs() -> tuple[tuple[list[torch.Tensor], ...], list[torch.Tensor]]:
    torch.manual_seed(1)
    f = [torch.randn(B, *SHAPE) for _ in range(T)]
    h = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    v = [torch.randn(B, 3, *SHAPE) for _ in range(T)]
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    return (f, h, v), g


def assert_output_shapes(out: tuple[list[torch.Tensor], ...]) -> None:
    f, h, v = out
    assert len(f) == len(h) == len(v) == T
    assert all(ft.shape == (B, *SHAPE) for ft in f)
    assert all(ht.shape == (B, *SINO_SHAPE) for ht in h)
    assert all(vt.shape == (B, 3, *SHAPE) for vt in v)


CONFIGS = [
    pytest.param(recurrent, shared_time, shared_iters, id=f"recurrent={recurrent}-shared_time={shared_time}-shared_iters={shared_iters}")
    for recurrent in (True, False)
    for shared_time in ((True, False) if recurrent else (False,))
    for shared_iters in (True, False)
]


@pytest.mark.parametrize("recurrent, shared_time, shared_iters", CONFIGS)
@pytest.mark.parametrize("grad_checkpointing", [False, True])
def test_unrolled_shapes_and_gradients(recurrent: bool, shared_time: bool, shared_iters: bool, grad_checkpointing: bool) -> None:
    model = make_unrolled(recurrent, shared_time, shared_iters, grad_checkpointing)
    x, g = make_inputs()

    out = model(x, g)
    assert_output_shapes(out)

    loss = sum(t.square().mean() for ts in out for t in ts)
    loss.backward()  # type: ignore[union-attr]
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"Parameters without gradient: {missing}"


@pytest.mark.parametrize("recurrent, shared_time, shared_iters", CONFIGS)
def test_checkpointing_matches_plain_forward_and_backward(recurrent: bool, shared_time: bool, shared_iters: bool) -> None:
    model = make_unrolled(recurrent, shared_time, shared_iters, grad_checkpointing=False)
    x, g = make_inputs()

    def run(grad_checkpointing: bool) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        model.zero_grad()
        model.grad_checkpointing = grad_checkpointing
        out = model(x, g)
        sum(t.square().mean() for ts in out for t in ts).backward()  # type: ignore[union-attr]
        return [t.detach() for ts in out for t in ts], {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    out_plain, grads_plain = run(False)
    out_ckpt, grads_ckpt = run(True)

    for a, b in zip(out_plain, out_ckpt):
        torch.testing.assert_close(a, b)
    assert grads_plain.keys() == grads_ckpt.keys()
    for name in grads_plain:
        torch.testing.assert_close(grads_plain[name], grads_ckpt[name], msg=name)


def test_unrolled_iterations_share_ray_trafo() -> None:
    model = make_unrolled(recurrent=True, shared_time=False, shared_iters=False, grad_checkpointing=False, num_iters=3)
    ray_trafo = model.hlpd_model_list[0].ray_trafo
    assert all(block.ray_trafo is ray_trafo for block in model.hlpd_model_list)
    # The nets themselves are still independent copies.
    assert model.hlpd_model_list[0].primal_net is not model.hlpd_model_list[1].primal_net


def make_hlpd_model(monkeypatch: pytest.MonkeyPatch, recurrent: bool, normalize: bool = False, intensity: str = "physical", zero_init: bool = True) -> HLPDModel:
    monkeypatch.setattr(hlpd, "RayTransform", MockRayTransform)
    torch.manual_seed(0)
    model = HLPDModel(
        unrolled_params=HLPDUnrolledParams(num_iters=2),
        block_params=HLPDBlockParams(recurrent=recurrent, time_steps=T),
        conv_stack_params=CONV_PARAMS,
        ray_trafo_params=RAY_PARAMS,
        deform_params=DeformParams(N=3),
        T=T,
        shape=SHAPE,
        normalization=NormalizationParams(operator=normalize, power_iters=200, intensity=intensity),
        mu_water=0.02,
        zero_init=zero_init,
    )
    meta = {"resampled_pixel_spacing": (1.0, 1.0), "resampled_slice_thickness": 2.0, "time_steps": [1.0] * T}
    model.set_geometry(meta)
    return model


@pytest.mark.parametrize("recurrent", [True, False])
def test_hlpd_model_forward(monkeypatch: pytest.MonkeyPatch, recurrent: bool) -> None:
    model = make_hlpd_model(monkeypatch, recurrent)
    assert model.deform.extent == model.extent == (8.0, 6.0, 6.0)

    # Only the unrolled model's copies of the nets are registered.
    assert {name.split(".")[0] for name, _ in model.named_parameters()} == {"unrolled_model"}

    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    f_template, v = model(g)

    assert f_template.shape == (B, *SHAPE)
    assert len(v) == T
    assert all(vt.shape == (B, *SHAPE, 3) for vt in v)


@pytest.mark.parametrize("domain", ["image", "projection"])
def test_hlpd_model_trains_with_lddmm_loss(monkeypatch: pytest.MonkeyPatch, domain: str) -> None:
    """One training step with the LDDMM loss in the image or projection domain reaches every parameter and lowers the loss."""
    model = make_hlpd_model(monkeypatch, recurrent=True)
    helmholtz_op = HelmholtzOperator(dim=3, extent=model.extent)
    forward_op = model.ray_trafo if domain == "projection" else None
    loss_fn = LDDMMloss(flow_deform_op=model.deform, helmholtz_op=helmholtz_op, lambda_reg=1e-3, forward_op=forward_op)  # type: ignore[arg-type]

    torch.manual_seed(1)
    x_true = [torch.rand(B, *SHAPE) for _ in range(T)]
    g = model.ray_trafo(x_true)
    y = g if domain == "projection" else x_true

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        f_template, v = model(g)
        loss, x_deformed = loss_fn(f_template, y, v)
        loss.total_loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"Parameters without gradient: {missing}"
        optimizer.step()
        losses.append(loss.total_loss.item())

    assert len(x_deformed) == T and all(xd.shape == (B, *SHAPE) for xd in x_deformed)
    assert losses[1] < losses[0]


def test_lddmm_loss_projection_domain_requires_time_dependent_input() -> None:
    deform = FlowDeformationOperator(N=3, integration="euler", shape=SHAPE, extent=(8.0, 6.0, 6.0))
    loss_fn = LDDMMloss(flow_deform_op=deform, helmholtz_op=HelmholtzOperator(dim=3), forward_op=make_ray_trafo())  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        loss_fn(torch.rand(B, *SHAPE), torch.rand(B, *SINO_SHAPE), torch.zeros(B, *SHAPE, 3))


def test_flow_deformation_normalizes_time_dependent_velocity_once() -> None:
    torch.manual_seed(0)
    deform = FlowDeformationOperator(N=3, integration="euler", shape=SHAPE, extent=(8.0, 6.0, 6.0))
    x = torch.randn(B, *SHAPE)
    v = [torch.randn(B, *SHAPE, 3) for _ in range(T)]

    out_list = deform(x, v)
    out_single = [deform(x, vt) for vt in v]

    for a, b in zip(out_list, out_single):
        torch.testing.assert_close(a, b)


def test_operator_norm_matches_spectral_norm() -> None:
    ray_trafo = make_ray_trafo()
    for t in range(T):
        norm = operator_norm(lambda x: ray_trafo.forward_bin(x, t), lambda y: ray_trafo.adjoint_bin(y, t), (1, *SHAPE), max_iters=500, rtol=1e-8)
        assert norm == pytest.approx(torch.linalg.matrix_norm(ray_trafo.ray_transform[t], ord=2).item(), rel=1e-3)


def test_normalized_operator_and_data_are_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    """With normalization, the operator in the iterations has norm 1 and maps x to the scaled data of the physical operator."""
    model = make_hlpd_model(monkeypatch, recurrent=True, normalize=True)
    block_ray_trafo = model.unrolled_model.hlpd_model_list[0].ray_trafo
    assert block_ray_trafo is model.block_ray_trafo and block_ray_trafo.ray_trafo is model.ray_trafo
    assert all(s != 1.0 for s in block_ray_trafo.scales)

    for t in range(T):
        norm = operator_norm(lambda x: block_ray_trafo.forward_bin(x, t), lambda y: block_ray_trafo.adjoint_bin(y, t), (1, *SHAPE), max_iters=500, rtol=1e-8)
        assert norm == pytest.approx(1.0, rel=5e-2)  # the model's estimate of ||T_t|| stops at a relative change of 1e-3, and power iteration converges from below

    x = [torch.rand(B, *SHAPE) for _ in range(T)]
    for a, b in zip(block_ray_trafo(x), model.scale_data(model.ray_trafo(x))):
        torch.testing.assert_close(a, b)


def test_operator_is_not_normalized_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_hlpd_model(monkeypatch, recurrent=True)
    assert model.block_ray_trafo.scales == [1.0] * T
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    assert all(a is b for a, b in zip(model.scale_data(g), g))


@pytest.mark.parametrize("recurrent", [True, False])
def test_block_is_identity_at_initialization(recurrent: bool) -> None:
    """With zero initialized final layers, the residual updates of an untrained block are zero."""
    block = make_unrolled(recurrent=recurrent, shared_time=False, shared_iters=False, grad_checkpointing=False).hlpd_model_list[0]
    torch.manual_seed(1)
    f = [torch.randn(B, *SHAPE) for _ in range(T)]
    h = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    v = [torch.randn(B, 3, *SHAPE) for _ in range(T)]
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    f_pred, h_pred, v_pred = block((f, h, v), g)
    for pred, init in [(f_pred, f), (h_pred, h), (v_pred, v)]:
        for a, b in zip(pred, init):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("intensity", ["physical", "water"])
@pytest.mark.parametrize("normalize", [False, True])
def test_untrained_model_returns_fbp_in_physical_units(monkeypatch: pytest.MonkeyPatch, intensity: str, normalize: bool) -> None:
    """The untrained model returns its FBP initialization, in mm^-1 whatever the scaling of the operator and the intensities."""
    model = make_hlpd_model(monkeypatch, recurrent=True, normalize=normalize, intensity=intensity, zero_init=False)
    assert model.intensity_scale == (50.0 if intensity == "water" else 1.0)
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    f_template, v = model(g)
    torch.testing.assert_close(f_template, model.ray_trafo.FBP(g)[0])
    assert all(torch.count_nonzero(vt) == 0 for vt in v)


def test_water_intensity_mode_needs_mu_water() -> None:
    with pytest.raises(AssertionError):
        HLPDModel(HLPDUnrolledParams(num_iters=1), HLPDBlockParams(time_steps=T), CONV_PARAMS, RAY_PARAMS, DeformParams(N=3), T=T, shape=SHAPE,
                  normalization=NormalizationParams(intensity="water"))
