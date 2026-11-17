"""
Tests of the adjoint derivative of the flow deformation operator (operators/lddmm/adjoint.py): the continuous formula against the exact adjoint
of the discretized operator (autograd), its convergence under grid refinement, and backpropagation through the approximations.
"""
import pytest
import torch

from operators.lddmm.deform import FlowDeformationOperator
from operators.lddmm.adjoint import adjoint_derivative, flow, smooth, spacing, spatial_gradient, warp

EXTENT = (72.0, 70.0, 70.0)


def problem(shape, backend: str, amp_mm: float, seed: int = 0):
    """A smooth image, a smooth velocity field of maximal norm amp_mm and a smooth f~, as continuous functions sampled on the grid, so
    that refining the grid refines the same problem."""
    g = torch.Generator().manual_seed(seed)
    deform = FlowDeformationOperator(N=6, shape=shape, extent=EXTENT, backend=backend, integrator_backend="grid_sample", order=3)
    z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, s, dtype=torch.float64) for s in shape], indexing="ij")
    deform.velocity_integrator.id = deform.velocity_integrator._identity_grid(shape).double()
    f = (torch.exp(-2 * ((x / 0.5) ** 2 + (y / 0.4) ** 2 + (z / 0.6) ** 2)) + 0.3 * torch.sin(3 * x) * torch.cos(2 * y))[None]
    a = torch.rand(6, generator=g, dtype=torch.float64)
    v = torch.stack([torch.sin(2 * x + a[0]) * torch.cos(y + a[1]), torch.cos(x + 2 * z + a[2]), torch.sin(y + z + a[3])])[None]
    v = v / v.norm(dim=1).max() * amp_mm
    f_tilde = (torch.cos(2 * x + a[4]) * torch.sin(3 * y + a[5]) * torch.exp(-z ** 2))[None]
    return deform, f, v, f_tilde


def rel(a, b, crop=3):
    I = (slice(None), slice(None), slice(crop, -crop), slice(crop, -crop), slice(crop, -crop))
    return ((a - b)[I].norm() / b[I].norm()).item()


@pytest.mark.parametrize("backend", ["grid_sample", "grid_pull"])
def test_zero_velocity_all_methods_are_minus_ftilde_grad_f(backend):
    deform, f, v, f_tilde = problem((12, 16, 16), backend, 0.0)
    expected = -f_tilde.unsqueeze(1) * spatial_gradient(f, spacing(deform, f.shape[1:]), "spline" if backend == "grid_pull" else "central")
    for method in ["right", "left", "midpoint", "trapezoid", "gauss3"]:
        assert rel(adjoint_derivative(deform, f, v, f_tilde, method), expected, crop=1) < 1e-4, method


def test_formula_matches_exact_adjoint_and_converges_under_refinement():
    """The continuous formula (*) by quadrature against the exact adjoint of the discrete operator, on a coarse and a twice finer grid
    of the same smooth problem: close, and closer on the finer grid (the difference is the discretization, not the formula)."""
    errs = []
    for shape in [(12, 20, 20), (24, 40, 40)]:
        deform, f, v, f_tilde = problem(shape, "grid_pull", 4.0)
        ref = adjoint_derivative(deform, f, v, f_tilde, "autograd")
        errs.append(rel(adjoint_derivative(deform, f, v, f_tilde, "gauss4"), ref, crop=shape[0] // 6))
    assert errs[0] < 0.1 and errs[1] < 0.6 * errs[0], errs


def test_quadrature_order():
    """One-point rules at the endpoints are first order in |v|, the midpoint rule second order (against the converged quadrature)."""
    errs = {}
    for amp in [2.0, 4.0]:
        deform, f, v, f_tilde = problem((16, 24, 24), "grid_pull", amp)
        ref = adjoint_derivative(deform, f, v, f_tilde, "gauss6")
        errs[amp] = {m: rel(adjoint_derivative(deform, f, v, f_tilde, m), ref) for m in ["right", "left", "midpoint"]}
    for m, order in [("right", 1), ("left", 1), ("midpoint", 2)]:
        ratio = errs[4.0][m] / errs[2.0][m]
        assert 0.7 * 2 ** order < ratio < 1.4 * 2 ** order, (m, errs)


@pytest.mark.parametrize("method", ["right", "midpoint"])
@pytest.mark.parametrize("backend", ["grid_sample", "grid_pull"])
def test_backpropagation_through_approximations(method, backend):
    deform, f, v, f_tilde = problem((8, 12, 12), backend, 2.0)
    f, v, f_tilde = (x.float().requires_grad_(True) for x in (f, v, f_tilde))
    deform.velocity_integrator.id = deform.velocity_integrator.id.float()
    adjoint_derivative(deform, f, v, f_tilde, method).square().sum().backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in (f, v, f_tilde))


def test_exact_adjoint_double_backward_needs_grid_sample():
    """The exact adjoint is differentiable with the grid_sample action, but not with grid_pull (torch-interpol has no double backward)."""
    deform, f, v, f_tilde = problem((8, 12, 12), "grid_sample", 2.0)
    v = v.float().requires_grad_(True)
    deform.velocity_integrator.id = deform.velocity_integrator.id.float()
    adjoint_derivative(deform, f.float(), v, f_tilde.float(), "autograd").square().sum().backward()
    assert v.grad is not None and v.grad.abs().sum() > 0

    deform, f, v, f_tilde = problem((8, 12, 12), "grid_pull", 2.0)
    v = v.float().requires_grad_(True)
    deform.velocity_integrator.id = deform.velocity_integrator.id.float()
    with pytest.raises(RuntimeError):
        adjoint_derivative(deform, f.float(), v, f_tilde.float(), "autograd").square().sum().backward()


def test_right_rule_is_gradient_of_compositive_perturbation():
    """The right rule is the exact gradient of u -> <W(f, v) o (id - u), f~> at u = 0. With the cubic interpolant: at the grid points,
    the derivative of the linear one is one-sided, which autograd picks and the central differences do not."""
    deform, f, v, f_tilde = problem((12, 16, 16), "grid_pull", 3.0)
    Wv = warp(deform, f, flow(deform, v, 1.0))
    u = torch.zeros_like(v, requires_grad=True)
    ident = FlowDeformationOperator(N=0, shape=f.shape[1:], extent=EXTENT, backend="grid_pull", order=3)
    ident.velocity_integrator.id = ident.velocity_integrator._identity_grid(f.shape[1:]).double()
    (g,) = torch.autograd.grad((warp(ident, Wv, flow(ident, u, 1.0)) * f_tilde).sum(), u)
    assert rel(adjoint_derivative(deform, f, v, f_tilde, "right"), g) < 0.05
