"""
Checks the interpolation backends of the deformation operators (see operators.lddmm.deform.resample):

  - grid_pull with linear interpolation is the same interpolation as grid_sample, for both paddings,
  - cubic grid_pull interpolates: the identity deformation returns the image, and a smooth image is resampled more accurately than with
    linear interpolation,
  - the grid_pull integrator integrates a translation fully (as test/test_velocity_integration.py for grid_sample),
  - FlowDeformationOperator runs forward and backward with every combination of backends, in 2D and 3D.

Run: pytest test/test_deform_backends.py
"""
import math

import pytest
import torch

from operators.lddmm.deform import FlowDeformationOperator, VelocityIntegrator, resample

SHAPE_3D = (12, 14, 16)
MARGIN = 3  # voxels excluded at each edge, more than the displacements below


def _identity(shape: tuple[int, ...]) -> torch.Tensor:
    return VelocityIntegrator(N=1, shape=shape).id.unsqueeze(0)  # type: ignore[union-attr]


def _smooth_displacement(shape: tuple[int, ...], amplitude: float = 0.08) -> torch.Tensor:
    """A smooth (1, *shape, C) displacement in normalized units, a fraction of a voxel to about a voxel."""
    grid = _identity(shape)
    u = torch.stack([amplitude * torch.sin(math.pi * grid[..., (c + 1) % len(shape)]) for c in range(len(shape))], dim=-1)
    return u


def _interior(ndim: int) -> tuple[slice, ...]:
    return (slice(None), slice(None)) + (slice(MARGIN, -MARGIN),) * ndim


@pytest.mark.parametrize("padding", ["zeros", "border"])
def test_linear_grid_pull_matches_grid_sample(padding: str):
    torch.manual_seed(0)
    x = torch.randn(2, 1, *SHAPE_3D)
    phi = (_identity(SHAPE_3D) + _smooth_displacement(SHAPE_3D)).expand(2, *SHAPE_3D, 3)

    ref = resample(x, phi, backend="grid_sample", padding=padding)
    out = resample(x, phi, backend="grid_pull", order=1, padding=padding)

    torch.testing.assert_close(out[_interior(3)], ref[_interior(3)], atol=1e-5, rtol=0)


@pytest.mark.parametrize("order", [3, [3, 1, 1]])
def test_cubic_identity_returns_the_image(order):
    torch.manual_seed(0)
    x = torch.randn(1, 1, *SHAPE_3D)

    out = resample(x, _identity(SHAPE_3D), backend="grid_pull", order=order)

    torch.testing.assert_close(out, x, atol=1e-4, rtol=0)


def test_cubic_is_more_accurate_than_linear_on_a_smooth_image():
    shape = (32, 32)
    grid = _identity(shape)
    f = lambda g: torch.sin(2 * math.pi * g[..., 0]) * torch.cos(math.pi * g[..., 1])  # noqa: E731
    x = f(grid).unsqueeze(1)
    phi = grid + _smooth_displacement(shape)
    exact = f(phi).unsqueeze(1)

    err = {name: (resample(x, phi, **kwargs) - exact)[_interior(2)].abs().max().item()
           for name, kwargs in {"linear": {"backend": "grid_sample"}, "cubic": {"backend": "grid_pull", "order": 3}}.items()}

    assert err["cubic"] < 0.1 * err["linear"], err


@pytest.mark.parametrize("integration", ["euler", "scaling_and_squaring"])
@pytest.mark.parametrize("order", [1, 3])
def test_grid_pull_integrator_integrates_a_translation(integration: str, order: int):
    vi = VelocityIntegrator(N=4, integration=integration, shape=SHAPE_3D, backend="grid_pull", order=order)
    v = torch.zeros(1, *SHAPE_3D, 3)
    v[..., 0], v[..., 1], v[..., 2] = 0.1, 0.0, -0.05

    phi = vi(v)

    # NOTE: as the displacement, not phi itself, is resampled, the translation is exact up to the boundary.
    torch.testing.assert_close(phi, vi.id - v, atol=1e-5, rtol=0)


@pytest.mark.parametrize("shape", [(16, 18), SHAPE_3D])
@pytest.mark.parametrize("backend,integrator_backend", [("grid_sample", "grid_sample"), ("grid_pull", "grid_sample"), ("grid_pull", "grid_pull")])
@pytest.mark.parametrize("action", ["geometric", "mass_preserving"])
def test_flow_deformation_operator_backends(shape, backend: str, integrator_backend: str, action: str):
    torch.manual_seed(0)
    extent = tuple(float(s) for s in shape)
    op = FlowDeformationOperator(N=3, action=action, shape=shape, extent=extent, backend=backend, integrator_backend=integrator_backend, order=3)
    x = torch.randn(1, *shape, requires_grad=True)
    v = [(0.5 * torch.randn(1, *shape, len(shape))).requires_grad_() for _ in range(2)]

    y = op(x, v)
    sum(yt.square().sum() for yt in y).backward()  # type: ignore[union-attr]

    assert len(y) == 2 and all(yt.shape == (1, 1, *shape) for yt in y)
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(vt.grad is not None and torch.isfinite(vt.grad).all() and vt.grad.abs().sum() > 0 for vt in v)


def test_grid_sample_is_the_default():
    """The default keeps the previous behavior."""
    torch.manual_seed(0)
    x = torch.randn(1, *SHAPE_3D)
    v = 0.5 * torch.randn(1, *SHAPE_3D, 3)
    extent = tuple(float(s) for s in SHAPE_3D)

    default = FlowDeformationOperator(N=3, shape=SHAPE_3D, extent=extent)(x, v)
    explicit = FlowDeformationOperator(N=3, shape=SHAPE_3D, extent=extent, backend="grid_sample", integrator_backend="grid_sample")(x, v)

    torch.testing.assert_close(default, explicit)
