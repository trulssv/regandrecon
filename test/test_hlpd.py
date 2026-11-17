"""
Shape, gradient-flow and checkpointing checks for model.hlpd, run against a small mock ray transform so
that they don't depend on the ODL/ASTRA backend. The mock has the same interface as
operators.tomo.ray_trafo.RayTransform (list-valued __call__/adjoint and the per-time-bin forward_bin/adjoint_bin)
and is a random dense linear operator per time bin, mapping B x D x H x W volumes to B x nAngles x nCol x nRow
projections, with odl's (nAngles, nCol, nRow) projection axis order.
"""
import copy
from dataclasses import asdict

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
    IterationUnits,
    NormalizedRayTransform,
    RayTrafoParams,
    net_channels,
)
from model.hlpd_loss import HLPDLoss, LossParams, fidelity
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
EXTENT = (8.0, 6.0, 6.0)


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

    def bins_for_complete_data(self) -> int:
        """As for 90 degree time bins of a parallel beam."""
        return 2

    def FBP(self, y: list[torch.Tensor]) -> list[torch.Tensor]:
        """A linear stand-in for the FBP, which is all that the scaling of the data needs."""
        return self.adjoint(y)


def make_ray_trafo() -> MockRayTransform:
    ray_trafo = MockRayTransform(nDetectorCols=RAY_PARAMS.nDetectorCols, nDetectorRows=RAY_PARAMS.nDetectorRows, nViews=RAY_PARAMS.nViews)
    ray_trafo._init_ray_transform(SHAPE, tuple(float(s) for s in SHAPE), [1.0] * T)
    return ray_trafo


def make_deform() -> FlowDeformationOperator:
    return FlowDeformationOperator(N=3, integration="euler", shape=SHAPE, extent=EXTENT)


def make_unrolled(recurrent: bool, shared_time: bool, shared_iters: bool, grad_checkpointing: bool, num_iters: int = 2,
                  zero_init_final: bool = True) -> HLPDUnrolled:
    torch.manual_seed(0)
    p_in, p_out, d_in, d_out, m_in, m_out = net_channels(recurrent, T)
    conv_params = {**vars(CONV_PARAMS), "zero_init_final": zero_init_final}
    block = HLPDBlock(
        ConvStack(p_in, p_out, **conv_params),
        ConvStack(d_in, d_out, **conv_params),
        ConvStack(m_in, m_out, **conv_params),
        make_ray_trafo(),  # type: ignore[arg-type]
        make_deform(),
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
    w = [torch.randn(B, *SHAPE) for _ in range(T)]
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    return (f, h, v, w), g


def assert_output_shapes(out: tuple[list[torch.Tensor], ...]) -> None:
    f, h, v, w = out
    assert len(f) == len(h) == len(v) == len(w) == T
    assert all(ft.shape == (B, *SHAPE) for ft in f)
    assert all(wt.shape == (B, *SHAPE) for wt in w)
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


def test_unrolled_iterations_share_ray_trafo_and_deform() -> None:
    model = make_unrolled(recurrent=True, shared_time=False, shared_iters=False, grad_checkpointing=False, num_iters=3)
    ray_trafo = model.hlpd_model_list[0].ray_trafo
    deform = model.hlpd_model_list[0].deform
    assert all(block.ray_trafo is ray_trafo for block in model.hlpd_model_list)
    assert all(block.deform is deform for block in model.hlpd_model_list)
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
    assert model.deform.extent == model.extent == EXTENT
    # The blocks warp with the model's deformation operator, so they see the extent set by set_geometry.
    assert all(block.deform is model.deform for block in model.unrolled_model.hlpd_model_list)

    # Only the unrolled model's copies of the nets are registered.
    assert {name.split(".")[0] for name, _ in model.named_parameters()} == {"unrolled_model"}

    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    w, f, v = model(g)

    assert len(w) == len(f) == len(v) == T
    assert all(wt.shape == (B, *SHAPE) for wt in w)
    assert all(ft.shape == (B, *SHAPE) for ft in f)
    assert all(vt.shape == (B, *SHAPE, 3) for vt in v)


@pytest.mark.parametrize("domain", ["image", "projection"])
@pytest.mark.parametrize("recurrent", [True, False])
def test_hlpd_model_trains_with_hlpd_loss(monkeypatch: pytest.MonkeyPatch, domain: str, recurrent: bool) -> None:
    """One training step with the loss in the image or projection domain reaches every parameter and lowers the loss."""
    model = make_hlpd_model(monkeypatch, recurrent=recurrent)
    helmholtz_op = HelmholtzOperator(dim=3, extent=model.extent)
    loss_fn = HLPDLoss(helmholtz_op, model.units, model.deform, legacy_params(domain=domain, lambda_coupling=1.0, lambda_registered=1.0, lambda_window=0.5, lambda_reg=1e-3))

    torch.manual_seed(1)
    x_true = [torch.rand(B, *SHAPE) for _ in range(T)]
    g = model.ray_trafo(x_true)
    y = g if domain == "projection" else x_true

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        w, f, v = model(g)
        loss = loss_fn(w, f, v, y)
        loss.total_loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"Parameters without gradient: {missing}"
        optimizer.step()
        losses.append(loss.total_loss.item())

    assert loss.coupling_loss.item() > 0 and loss.registered_loss.item() > 0 and loss.window_loss.item() > 0
    assert losses[1] < losses[0]


def legacy_params(**kwargs) -> LossParams:
    """LossParams from the flat config of earlier runs (see LossParams.from_dict), with lambda_reg = 0.1 unless given."""
    return LossParams.from_dict({"lambda_reg": 0.1, **kwargs})


def image_loss(params: LossParams) -> HLPDLoss:
    """The loss in physical units (with the 2 time bins for complete data of the mock ray transform)."""
    units = IterationUnits(1.0, NormalizedRayTransform(make_ray_trafo(), T))  # type: ignore[arg-type]
    return HLPDLoss(HelmholtzOperator(dim=3, extent=EXTENT), units, make_deform(), params)


def make_image_loss(**kwargs) -> HLPDLoss:
    """The loss in the image domain from the flat config of earlier runs, see legacy_params."""
    return image_loss(legacy_params(domain="image", **kwargs))


def test_hlpd_loss_terms() -> None:
    """The coupling term averages the predictions w_t and the next time bins moved back against the reconstructions, and each term is
    weighted in the total."""
    torch.manual_seed(0)
    f = [torch.rand(B, *SHAPE) for _ in range(T)]
    y = [torch.rand(B, *SHAPE) for _ in range(T)]
    v = [torch.randn(B, *SHAPE, 3) for _ in range(T)]
    loss_fn = make_image_loss(lambda_coupling=2.0, lambda_registered=0.25, lambda_window=0.125, window_k=1, lambda_reg=0.5)

    backward = sum(torch.mean((moved[0] - f[t]) ** 2) for t, moved in enumerate(loss_fn._moved_back(f, v, 1)))
    matched = loss_fn(f, f, v, y)
    torch.testing.assert_close(matched.coupling_loss, backward / 2)

    w = [ft + 0.1 for ft in f]
    loss = loss_fn(w, f, v, y)
    torch.testing.assert_close(loss.coupling_loss, (T * 0.01 + backward) / 2)
    torch.testing.assert_close(loss.data_loss, torch.stack([torch.mean((ft - yt) ** 2) for ft, yt in zip(f, y)]).sum())
    # One gap: the predictions w_t against the targets of their time bins.
    torch.testing.assert_close(loss.registered_loss, torch.stack([torch.mean((wt - yt) ** 2) for wt, yt in zip(w, y)]).sum())
    # The window of the 3 time bins around t, against each of their targets.
    fbar = [(f[t - 1] + f[t] + f[(t + 1) % T]) / 3 for t in range(T)]
    expected_window = sum(torch.mean((fbar[t] - y[(t + i) % T]) ** 2) for t in range(T) for i in (-1, 0, 1)) / 3
    torch.testing.assert_close(loss.window_loss, expected_window)
    torch.testing.assert_close(loss.total_loss, loss.data_loss + 2.0 * loss.coupling_loss + 0.25 * loss.registered_loss
                               + 0.125 * loss.window_loss + 0.5 * loss.regularization_loss)


def fidelity_reference(a: torch.Tensor, b: torch.Tensor, kind: str, delta: float) -> torch.Tensor:
    r = (a - b).abs()
    if kind == "l1":
        return r.mean()
    if kind == "huber":
        return torch.where(r < delta, r ** 2, 2 * delta * r - delta ** 2).mean()
    return (r ** 2).mean()


@pytest.mark.parametrize("kind", ["mse", "l1", "huber"])
def test_fidelity(kind: str) -> None:
    """The Huber loss equals the squared error for residuals below huber_delta and grows linearly above it."""
    torch.manual_seed(0)
    a, b = torch.randn(1000), torch.randn(1000)
    torch.testing.assert_close(fidelity(a, b, kind, 0.5), fidelity_reference(a, b, kind, 0.5))


@pytest.mark.parametrize("kind", ["l1", "huber"])
def test_each_term_has_its_own_fidelity(kind: str) -> None:
    """The fidelity of one term does not change the others."""
    torch.manual_seed(0)
    f = [torch.rand(B, *SHAPE) for _ in range(T)]
    y = [ft + torch.randn(B, *SHAPE) for ft in f]
    v = [torch.randn(B, *SHAPE, 3) for _ in range(T)]
    w = [ft + 0.3 * torch.randn(B, *SHAPE) for ft in f]

    def loss(**terms):
        params = LossParams(domain="image", **{"registered": {"include": True}, "window": {"include": True}, **terms})
        return image_loss(params)(w, f, v, y)

    base = loss()
    for name in ("data", "coupling", "registered", "window"):
        changed = loss(**{name: {"include": True, "fidelity": kind, "huber_delta": 0.5}})
        for other in ("data", "coupling", "registered", "window"):
            same = torch.isclose(getattr(changed, f"{other}_loss"), getattr(base, f"{other}_loss"))
            assert bool(same) == (other != name), (name, other)
    data = loss(data={"fidelity": kind, "huber_delta": 0.5})
    torch.testing.assert_close(data.data_loss, torch.stack([fidelity_reference(ft, yt, kind, 0.5) for ft, yt in zip(f, y)]).sum())


def test_excluded_terms_are_zero_and_weights_apply() -> None:
    torch.manual_seed(0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T)], [torch.rand(B, *SHAPE) for _ in range(T)]
    v = [torch.randn(B, *SHAPE, 3) for _ in range(T)]
    params = LossParams(domain="image", data={"weight": 2.0}, coupling={"include": False}, regularization={"include": False})
    loss = image_loss(params)(f, f, v, y)
    assert loss.coupling_loss.item() == loss.regularization_loss.item() == loss.registered_loss.item() == loss.window_loss.item() == 0.0
    torch.testing.assert_close(loss.total_loss, 2.0 * loss.data_loss)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("norm", ["l2", "helmholtz"])
def test_temporal_regularization(order: int, norm: str) -> None:
    """The cyclic differences of the velocities over the time bins, zero for velocities constant in time, and weighted in the total."""
    torch.manual_seed(0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T)], [torch.rand(B, *SHAPE) for _ in range(T)]
    v = [torch.randn(B, *SHAPE, 3) for _ in range(T)]
    params = LossParams(domain="image", coupling={"include": False}, regularization={"include": False},
                        temporal={"include": True, "weight": 0.5, "order": order, "norm": norm})
    loss_fn = image_loss(params)

    constant = loss_fn(f, f, [v[0]] * T, y)
    torch.testing.assert_close(constant.temporal_loss, torch.zeros(()))

    loss = loss_fn(f, f, v, y)
    M = loss_fn.helmholtz_op if norm == "helmholtz" else (lambda d: d)
    diff = (lambda t: v[(t + 1) % T] - v[t]) if order == 1 else (lambda t: v[(t + 1) % T] - 2 * v[t] + v[(t - 1) % T])
    torch.testing.assert_close(loss.temporal_loss, sum(M(diff(t)).square().mean() for t in range(T)))
    torch.testing.assert_close(loss.total_loss, loss.data_loss + 0.5 * loss.temporal_loss)


def test_temporal_regularization_is_excluded_by_default() -> None:
    torch.manual_seed(0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T)], [torch.rand(B, *SHAPE) for _ in range(T)]
    v = [torch.randn(B, *SHAPE, 3) for _ in range(T)]
    assert make_image_loss()(f, f, v, y).temporal_loss.item() == 0.0


def test_supervised_motion_term() -> None:
    """The ground truth volumes moved by the motion against those of the next time bins: without motion, each against the next one."""
    torch.manual_seed(0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T)], [torch.rand(B, *SHAPE) for _ in range(T)]
    params = LossParams(domain="image", coupling={"include": False}, regularization={"include": False},
                        supervised_motion={"include": True, "weight": 0.5})
    loss_fn = image_loss(params)
    zero = [torch.zeros(B, *SHAPE, 3) for _ in range(T)]
    loss = loss_fn(f, f, zero, y)
    torch.testing.assert_close(loss.supervised_motion_loss, sum(torch.mean((y[t - 1] - y[t]) ** 2) for t in range(T)))
    torch.testing.assert_close(loss.total_loss, loss.data_loss + 0.5 * loss.supervised_motion_loss)

    # The term reaches the motion.
    v = [torch.zeros(B, *SHAPE, 3, requires_grad=True) for _ in range(T)]
    loss_fn(f, f, v, y).supervised_motion_loss.backward()
    assert all(vt.grad is not None and vt.grad.abs().sum() > 0 for vt in v)
    assert make_image_loss()(f, f, zero, y).supervised_motion_loss.item() == 0.0


def test_supervised_motion_term_needs_the_image_domain() -> None:
    with pytest.raises(AssertionError, match="image domain"):
        LossParams(domain="projection", supervised_motion={"include": True})


def test_set_params_changes_the_terms() -> None:
    torch.manual_seed(0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T)], [torch.rand(B, *SHAPE) for _ in range(T)]
    v = [torch.randn(B, *SHAPE, 3) for _ in range(T)]
    loss_fn = image_loss(LossParams(domain="image"))
    assert loss_fn(f, f, v, y).regularization_loss.item() > 0
    loss_fn.set_params(LossParams(domain="image", data={"weight": 3.0}, regularization={"include": False}))
    loss = loss_fn(f, f, v, y)
    assert loss.regularization_loss.item() == 0.0
    torch.testing.assert_close(loss.total_loss, 3.0 * loss.data_loss + loss.coupling_loss)
    with pytest.raises(AssertionError, match="domain"):
        loss_fn.set_params(LossParams(domain="projection"))
    with pytest.raises(AssertionError, match="Helmholtz"):
        loss_fn.set_params(LossParams(domain="image", regularization={"helmholtz": [2.0, 1.0, 1.0]}))


@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize("groups", [("dual", "primal"), ("motion",), ("dual", "primal", "motion")])
def test_set_trainable_freezes_the_other_nets(monkeypatch: pytest.MonkeyPatch, recurrent: bool, groups: tuple[str, ...]) -> None:
    """Only the nets of the trained groups get gradients, in every block."""
    model = make_hlpd_model(monkeypatch, recurrent=recurrent)
    model.set_trainable(groups)
    loss_fn = HLPDLoss(HelmholtzOperator(dim=3, extent=model.extent), model.units, model.deform, LossParams(domain="image"))
    torch.manual_seed(1)
    x_true = [torch.rand(B, *SHAPE) for _ in range(T)]
    w, f, v = model(model.ray_trafo(x_true))
    loss_fn(w, f, v, x_true).total_loss.backward()
    for name, p in model.named_parameters():
        trained = any(f"{group}_net" in name for group in groups)
        assert p.requires_grad == trained and (p.grad is not None) == trained, name
    model.set_trainable(model.NET_GROUPS)
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(AssertionError, match="Unknown nets"):
        model.set_trainable(["sigma"])


@pytest.mark.parametrize("recurrent", [True, False])
def test_skip_motion_matches_running_the_zero_motion_nets(monkeypatch: pytest.MonkeyPatch, recurrent: bool) -> None:
    """With frozen, untrained motion nets, skipping them and the warps gives the same outputs and gradients as running them with
    exact identity warps (that the warp at v = 0 is the identity, see test_warp_with_zero_velocity_is_the_identity).
    NOTE: compared with the real warps instead, the gradients of this small random recurrent model differ by up to 7% in float32 (2e-5
    in float64), as it amplifies the rounding of the warps (relative 1e-6) by orders of magnitude."""
    model = make_hlpd_model(monkeypatch, recurrent=recurrent)
    # Move the reconstruction nets away from their zero initialization, so that they depend on their inputs (and on w).
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.1)
    for block in model.unrolled_model.hlpd_model_list:
        for net in (block.motion_net if recurrent else [block.motion_net]):
            torch.nn.init.zeros_(net.blocks[-1].conv.weight)
            torch.nn.init.zeros_(net.blocks[-1].conv.bias)
    model.set_trainable(("dual", "primal"))
    assert model.skip_motion
    loss_fn = HLPDLoss(HelmholtzOperator(dim=3, extent=model.extent), model.units, model.deform,
                       LossParams(domain="image", coupling={"include": False}, regularization={"include": False}))
    torch.manual_seed(1)
    x_true = [torch.rand(B, *SHAPE) for _ in range(T)]
    g = model.ray_trafo(x_true)

    def run() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        w, f, v = model(g)
        loss_fn(w, f, v, x_true).total_loss.backward()
        return w, f, v, {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    w_skip, f_skip, v_skip, grads_skip = run()
    monkeypatch.setattr(HLPDBlock, "_warp", lambda self, f, v: f + 0 * v.sum())  # identity, still connected to v
    for block in model.unrolled_model.hlpd_model_list:
        block.skip_motion = False
    w, f, v, grads = run()

    assert all(not vt.any() for vt in v_skip + v)
    for a, b in zip(f_skip + w_skip, f + w):
        torch.testing.assert_close(a, b)
    assert grads_skip.keys() == grads.keys() and grads and not any("motion_net" in name for name in grads)
    for name in grads:
        torch.testing.assert_close(grads_skip[name], grads[name], msg=name)


def test_warp_with_zero_velocity_is_the_identity() -> None:
    """W(f, 0) = f and its adjoint too, up to the rounding of the spline prefilter, so that skipping the warps of zero motion
    (HLPDBlock.skip_motion) changes neither the outputs nor the gradients."""
    shape = (16, 24, 24)
    deform = FlowDeformationOperator(**{**asdict(DeformParams(N=5)), "extent": tuple(float(n) for n in shape)}, shape=shape)
    torch.manual_seed(0)
    f = torch.rand(B, *shape, requires_grad=True)
    r = torch.randn(B, *shape)
    w = deform(f, torch.zeros(B, *shape, 3)).squeeze(1)
    (w * r).sum().backward()
    assert f.grad is not None
    assert (w - f).norm() / f.norm() < 1e-4
    assert (f.grad - r).norm() / r.norm() < 1e-4


def test_skip_motion_only_with_frozen_zero_motion_nets(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_hlpd_model(monkeypatch, recurrent=True)
    assert not model.skip_motion
    model.set_trainable(("dual", "primal"))
    assert model.skip_motion
    model.set_trainable(model.NET_GROUPS)
    assert not model.skip_motion
    # Trained motion nets (e.g. a reconstruction stage started from a later checkpoint) output motion, so they are run even if frozen.
    with torch.no_grad():
        model.unrolled_model.hlpd_model_list[-1].motion_net[0].blocks[-1].conv.bias.fill_(0.1)
    model.set_trainable(("dual", "primal"))
    assert not model.skip_motion


def test_loss_params_from_legacy_config() -> None:
    """The flat config of earlier runs maps to the same terms, with a zero weight excluding the registered and window terms."""
    params = LossParams.from_dict({"domain": "image", "lambda_coupling": 5.0, "coupling_gaps": 3, "lambda_registered": 0.0, "lambda_window": 0.5,
                                   "window_k": 2, "lambda_reg": 1e-3, "helmholtz": [1.0, 2.0, 3.0], "data_loss": "huber", "huber_delta": 0.2})
    assert params.coupling.include and params.coupling.weight == 5.0 and params.coupling.gaps == 3 and params.coupling.fidelity == "mse"
    assert not params.registered.include
    assert params.window.include and params.window.weight == 0.5 and params.window.k == 2 and params.window.fidelity == "huber"
    assert params.data.fidelity == "huber" and params.data.huber_delta == 0.2
    assert params.regularization.weight == 1e-3 and params.regularization.helmholtz == (1.0, 2.0, 3.0)
    assert LossParams.from_dict(asdict(params)) == params


def test_switched_off_terms_are_zero() -> None:
    torch.manual_seed(0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T)], [torch.rand(B, *SHAPE) for _ in range(T)]
    loss = make_image_loss(lambda_registered=0.0, lambda_window=0.0)(f, f, [torch.zeros(B, *SHAPE, 3) for _ in range(T)], y)
    assert loss.registered_loss.item() == 0.0 and loss.window_loss.item() == 0.0


def test_registered_term_over_more_time_bins_without_motion() -> None:
    """Without motion, the term with J = 2 gaps compares w_t and f_{t-2} with the target of time bin t, and averages over the gaps."""
    torch.manual_seed(0)
    f, w, y = ([torch.rand(B, *SHAPE) for _ in range(T)] for _ in range(3))
    v = [torch.zeros(B, *SHAPE, 3) for _ in range(T)]
    loss_fn = make_image_loss(lambda_registered=1.0, registered_gaps=2)
    assert loss_fn.n_registered_gaps() == 2 and make_image_loss().n_registered_gaps() == 1 and make_image_loss().n_coupling_gaps() == 1   # the mock ray transform needs 2 time bins for complete data

    loss = loss_fn(w, f, v, y)

    expected = sum(torch.mean((w[t] - y[t]) ** 2) + torch.mean((f[t - 2] - y[t]) ** 2) for t in range(T)) / 2
    torch.testing.assert_close(loss.registered_loss, expected, atol=1e-5, rtol=1e-5)


def test_coupling_term_over_more_time_bins_without_motion() -> None:
    """Without motion, the coupling with K = 2 gaps compares f_t with w_t, f_{t-2}, f_{t+1} and f_{t+2}, and averages over the gaps."""
    torch.manual_seed(0)
    T5 = 5  # 2K + 1 time bins
    f, w, y = ([torch.rand(B, *SHAPE) for _ in range(T5)] for _ in range(3))
    v = [torch.zeros(B, *SHAPE, 3) for _ in range(T5)]
    loss_fn = make_image_loss(coupling_gaps=2)
    assert loss_fn.n_coupling_gaps() == 2

    loss = loss_fn(w, f, v, y)

    expected = sum(torch.mean((w[t] - f[t]) ** 2) + sum(torch.mean((f[(t + j) % T5] - f[t]) ** 2) for j in (-2, 1, 2)) for t in range(T5)) / 4
    torch.testing.assert_close(loss.coupling_loss, expected, atol=1e-5, rtol=1e-5)
    assert loss.registered_loss.item() == 0.0   # off by default


def test_coupling_term_needs_2K_plus_1_time_bins() -> None:
    f = [torch.rand(B, *SHAPE) for _ in range(T)]
    with pytest.raises(AssertionError, match="2K_c"):
        make_image_loss(coupling_gaps=2)(f, f, [torch.zeros(B, *SHAPE, 3) for _ in range(T)], f)


def test_coupling_term_reaches_every_motion_on_the_way() -> None:
    torch.manual_seed(0)
    T5 = 5  # 2K + 1 time bins
    loss_fn = make_image_loss(coupling_gaps=2, lambda_reg=0.0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T5)], [torch.rand(B, *SHAPE) for _ in range(T5)]
    v = [(0.3 * torch.randn(B, *SHAPE, 3)).requires_grad_() for _ in range(T5)]
    # NOTE: w_t is detached, so that the motions only get gradients from the warps over two time bins.
    w = [loss_fn.deform(f[t - 1], v[t - 1]).squeeze(1).detach() for t in range(T5)]

    loss_fn(w, f, v, y).coupling_loss.backward()

    assert all(vt.grad is not None and vt.grad.abs().sum() > 0 for vt in v)


def test_registered_term_composes_the_motions() -> None:
    """The reconstruction of time bin t-2 is moved to time bin t by v_{t-2} and then v_{t-1}: one warp with the composed grid matches
    two warps in a row up to their interpolation, unlike the composition in the other order (the two shears do not commute), and the
    grid of a single motion warps exactly as the deformation operator."""
    shape, extent = (12, 24, 24), (24.0, 48.0, 48.0)
    deform = FlowDeformationOperator(N=3, integration="euler", shape=shape, extent=extent)
    loss_fn = make_image_loss()
    loss_fn.deform = deform
    z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in shape], indexing="ij")
    f = (torch.sin(3 * y) * torch.cos(2 * x) * torch.cos(z))[None]
    zero = torch.zeros_like(x)
    v1 = torch.stack([3.0 * torch.sin(2 * y), zero, zero], dim=-1)[None]   # displacement along x (mm) varying with y
    v2 = torch.stack([zero, 3.0 * torch.sin(2 * x), zero], dim=-1)[None]   # displacement along y varying with x
    compose = deform.velocity_integrator._compose

    torch.testing.assert_close(deform.group_action(f, loss_fn._grid(v1)), deform(f, v1))

    sequential = deform(deform(f, v1).squeeze(1), v2).squeeze(1)
    composed = deform.group_action(f, compose(loss_fn._grid(v1), loss_fn._grid(v2))).squeeze(1)
    other_order = deform.group_action(f, compose(loss_fn._grid(v2), loss_fn._grid(v1))).squeeze(1)
    interior = (slice(None), slice(2, -2), slice(2, -2), slice(2, -2))
    assert (composed - sequential)[interior].abs().max() < 0.01
    assert (other_order - sequential)[interior].abs().max() > 0.05
    assert (composed - f)[interior].abs().max() > 0.2   # the motion is not negligible


def test_coupling_term_moves_the_next_time_bins_back() -> None:
    """A sequence moved forward by v_0 and then v_1 is moved back to time bin 0 from both later time bins, up to interpolation. From time
    bin 2, one warp with the composed grid matches moving back by v_1 and then v_0 in a row, unlike the composition in the other order
    (the two shears do not commute)."""
    shape, extent = (12, 24, 24), (24.0, 48.0, 48.0)
    deform = FlowDeformationOperator(N=3, integration="euler", shape=shape, extent=extent)
    loss_fn = make_image_loss()
    loss_fn.deform = deform
    z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in shape], indexing="ij")
    f0 = (torch.sin(3 * y) * torch.cos(2 * x) * torch.cos(z))[None]
    zero = torch.zeros_like(x)
    v0 = torch.stack([3.0 * torch.sin(2 * y), zero, zero], dim=-1)[None]   # displacement along x (mm) varying with y
    v1 = torch.stack([zero, 3.0 * torch.sin(2 * x), zero], dim=-1)[None]   # displacement along y varying with x
    f1 = deform(f0, v0).squeeze(1)
    f2 = deform(f1, v1).squeeze(1)

    moved_back = loss_fn._moved_back([f0, f1, f2], [v0, v1, torch.zeros_like(v0)], 2)[0]
    back = lambda f, v: deform.group_action(f, loss_fn._inverse_grid(v)).squeeze(1)  # noqa: E731
    sequential = back(back(f2, v1), v0)
    compose = deform.velocity_integrator._compose
    wrong_order = deform.group_action(f2, compose(loss_fn._inverse_grid(v0), loss_fn._inverse_grid(v1))).squeeze(1)

    interior = (slice(None), slice(2, -2), slice(2, -2), slice(2, -2))
    assert (f2 - f0)[interior].abs().max() > 0.2   # the motion is not negligible
    # NOTE: the round trips interpolate f_0 up to three times with the linear interpolation of grid_sample, hence the looser tolerance.
    for j, moved in enumerate(moved_back, start=1):
        assert (moved - f0)[interior].abs().max() < 0.03, f"j = {j}: {(moved - f0)[interior].abs().max():.4f}"
    assert (moved_back[1] - sequential)[interior].abs().max() < 0.01
    assert (wrong_order - sequential)[interior].abs().max() > 0.05


def test_registered_term_reaches_every_motion_on_the_way() -> None:
    torch.manual_seed(0)
    loss_fn = make_image_loss(lambda_registered=1.0, registered_gaps=2, lambda_coupling=0.0, lambda_reg=0.0)
    f, y = [torch.rand(B, *SHAPE) for _ in range(T)], [torch.rand(B, *SHAPE) for _ in range(T)]
    v = [(0.3 * torch.randn(B, *SHAPE, 3)).requires_grad_() for _ in range(T)]
    w = [loss_fn.deform(f[t - 1], v[t - 1]).squeeze(1) for t in range(T)]

    loss_fn(w, f, v, y).registered_loss.backward()

    assert all(vt.grad is not None and vt.grad.abs().sum() > 0 for vt in v)


@pytest.mark.parametrize("domain", ["image", "projection"])
@pytest.mark.parametrize("normalize, intensity", [(False, "physical"), (True, "water")])
def test_hlpd_loss_uses_the_units_of_the_model(monkeypatch: pytest.MonkeyPatch, domain: str, normalize: bool, intensity: str) -> None:
    """The loss is computed in the units of the iterations: images scaled by the intensity scale c, and data by c / ||T_t||. Without
    normalization, these are the physical units."""
    model = make_hlpd_model(monkeypatch, recurrent=True, normalize=normalize, intensity=intensity)
    loss_fn = HLPDLoss(HelmholtzOperator(dim=3, extent=model.extent), model.units, model.deform, legacy_params(domain=domain, lambda_coupling=1.0, lambda_registered=1.0, lambda_reg=0.0))
    torch.manual_seed(0)
    x_true = [torch.rand(B, *SHAPE) for _ in range(T)]
    f = [torch.rand(B, *SHAPE) for _ in range(T)]
    w = [torch.rand(B, *SHAPE) for _ in range(T)]
    v = [torch.zeros(B, *SHAPE, 3) for _ in range(T)]
    g = model.ray_trafo(x_true)

    loss = loss_fn(w, f, v, g if domain == "projection" else x_true)

    c = model.intensity_scale
    s = [1.0 / n for n in model.operator_norms] if normalize else [1.0] * T
    if domain == "projection":
        expected_data = sum(torch.mean((c * st * (model.ray_trafo.forward_bin(ft, t) - gt)) ** 2) for t, (ft, gt, st) in enumerate(zip(f, g, s)))
    else:
        expected_data = sum(torch.mean((c * (ft - xt)) ** 2) for ft, xt in zip(f, x_true))
    torch.testing.assert_close(loss.data_loss, expected_data)
    # Without motion, the next time bin moved back is f_{t+1} itself.
    expected_coupling = sum(torch.mean((c * (wt - ft)) ** 2) + torch.mean((c * (f[(t + 1) % T] - ft)) ** 2) for t, (wt, ft) in enumerate(zip(w, f))) / 2
    torch.testing.assert_close(loss.coupling_loss, expected_coupling)
    if domain == "projection":
        expected_registered = sum(torch.mean((c * st * (model.ray_trafo.forward_bin(wt, t) - gt)) ** 2) for t, (wt, gt, st) in enumerate(zip(w, g, s)))
        torch.testing.assert_close(loss.registered_loss, expected_registered)
    # The model feeds its iterations the data in the same units.
    for a, b in zip(model.units.data(g), model.scale_data([c * gt for gt in g])):
        torch.testing.assert_close(a, b)


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
    (f, h, v, w), g = make_inputs()
    f_pred, h_pred, v_pred, w_pred = block((f, h, v, w), g)
    for pred, init in [(f_pred, f), (h_pred, h), (v_pred, v)]:
        for a, b in zip(pred, init):
            torch.testing.assert_close(a, b)
    # The predictions are always updated, cyclically: w_{t+1} = W(f_t, v_t), and w_0 = W(f_{T-1}, v_{T-1}).
    for t in range(T):
        torch.testing.assert_close(w_pred[(t + 1) % T], block._warp(f[t], v[t]))


@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize("grad_checkpointing", [False, True])
def test_predictions_are_consistent_with_the_output(recurrent: bool, grad_checkpointing: bool) -> None:
    """The returned predictions are the warps of the returned reconstructions with the returned velocities, so the loss can use them as they are."""
    model = make_unrolled(recurrent, shared_time=False, shared_iters=False, grad_checkpointing=grad_checkpointing, zero_init_final=False)
    x, g = make_inputs()
    f, _, v, w = model(x, g)
    block = model.hlpd_model_list[-1]
    for t in range(T):
        torch.testing.assert_close(w[(t + 1) % T], block._warp(f[t], v[t]))


def test_recurrent_primal_net_sees_the_fresh_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """In the recurrent block, the primal net of time bin t > 0 gets the prediction W(f_{t-1}, v_{t-1}) of the current iteration, and
    that of time bin 0 the prediction of the previous iteration."""
    block = make_unrolled(recurrent=True, shared_time=False, shared_iters=False, grad_checkpointing=False, zero_init_final=False).hlpd_model_list[0]
    seen: list[torch.Tensor] = []
    for net in block.primal_net:
        net.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0][:, 3].detach().clone()))
    (f, h, v, w), g = make_inputs()

    f_pred, _, v_pred, _ = block((f, h, v, w), g)

    torch.testing.assert_close(seen[0], w[0])
    for t in range(1, T):
        torch.testing.assert_close(seen[t], block._warp(f_pred[t - 1], v_pred[t - 1]))


@pytest.mark.parametrize("intensity", ["physical", "water"])
@pytest.mark.parametrize("normalize", [False, True])
def test_untrained_model_returns_fbp_in_physical_units(monkeypatch: pytest.MonkeyPatch, intensity: str, normalize: bool) -> None:
    """The untrained model returns its FBP initialization, in mm^-1 whatever the scaling of the operator and the intensities."""
    model = make_hlpd_model(monkeypatch, recurrent=True, normalize=normalize, intensity=intensity, zero_init=False)
    assert model.intensity_scale == (50.0 if intensity == "water" else 1.0)
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    w, f, v = model(g)
    fbp = model.ray_trafo.FBP(g)
    for t in range(T):
        torch.testing.assert_close(f[t], fbp[t])
        torch.testing.assert_close(w[t], fbp[t - 1], atol=1e-5, rtol=1e-5)  # the identity warp of the previous time bin, cyclically
    assert all(torch.count_nonzero(vt) == 0 for vt in v)


@pytest.mark.parametrize("intensity", ["physical", "water"])
def test_untrained_model_returns_given_fbp(monkeypatch: pytest.MonkeyPatch, intensity: str) -> None:
    """A given FBP (as stored by the data pipeline, in mm^-1) is used for the initialization instead of computing it, and is ignored
    with zero initialization."""
    model = make_hlpd_model(monkeypatch, recurrent=True, normalize=True, intensity=intensity, zero_init=False)
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    fbp = [torch.randn(B, *SHAPE) for _ in range(T)]
    _, f, _ = model(g, fbp)
    for ft, fbpt in zip(f, fbp):
        torch.testing.assert_close(ft, fbpt)

    zero_model = make_hlpd_model(monkeypatch, recurrent=True, normalize=True, intensity=intensity, zero_init=True)
    w, f, _ = zero_model(g, fbp)
    assert all(torch.count_nonzero(ft) == 0 and torch.count_nonzero(wt) == 0 for ft, wt in zip(f, w))


def test_water_intensity_mode_needs_mu_water() -> None:
    with pytest.raises(AssertionError):
        HLPDModel(HLPDUnrolledParams(num_iters=1), HLPDBlockParams(time_steps=T), CONV_PARAMS, RAY_PARAMS, DeformParams(N=3), T=T, shape=SHAPE,
                  normalization=NormalizationParams(intensity="water"))


@pytest.mark.parametrize("recurrent", [True, False])
def test_every_region_is_a_key_step_of_the_timing_plot(monkeypatch: pytest.MonkeyPatch, recurrent: bool) -> None:
    """Every region the model and its loss emit (with all loss terms switched on) is a key step of the timing plot, apart from the
    structural regions (iterations, time steps) and those nested in a key step, so that new operators show up in the plots."""
    from tracking.timeline import FORWARD_KEYS, StepTimeline, region

    model = make_hlpd_model(monkeypatch, recurrent=recurrent)
    loss_fn = HLPDLoss(HelmholtzOperator(dim=3, extent=model.extent), model.units, model.deform,
                       legacy_params(domain="projection", lambda_registered=1.0, registered_gaps=2, lambda_window=1.0))
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    timeline = StepTimeline(torch.device("cpu"))
    with timeline.record():
        with region("step/forward"):
            w, f, v = model(g)
        with region("step/loss"):
            loss_fn(w, f, v, g)

    labels = {e.label for e in timeline.events}
    structural = {lbl for lbl in labels if lbl.startswith("step/") or lbl == "hlpd/unrolled" or lbl.startswith("hlpd/iter")}
    nested = {e.label for e in timeline.events if any(parent in FORWARD_KEYS for parent in e.parents)}
    assert labels - structural - nested <= set(FORWARD_KEYS), f"Regions missing from FORWARD_KEYS: {labels - structural - nested - set(FORWARD_KEYS)}"
    assert {"hlpd/deform", "loss/data", "loss/coupling", "loss/registered", "loss/window", "loss/helmholtz"} <= set(timeline.history[-1])
    # The nested regions are not counted twice.
    assert nested and not nested & set(timeline.history[-1])
