r"""
The adjoint Gateaux derivative of the flow deformation operator, the input of the LDDMM-styled registration modules (model/motion.py).
Tested in test/test_lddmm_adjoint.py and benchmarked in test/bench_lddmm_adjoint.py.

The deformation operator of the HLPD model, W(f, v) = f o exp(v)^{-1} = f o phi_{1,0} with a stationary velocity field v per time bin
(operators/lddmm/deform.py), has the Gateaux derivative with respect to v (a time-constant perturbation u)

    dW|_f(v) u = - int_0^1 < grad(f o phi_{tau,0}) o phi_{1,tau}, u o phi_{1,tau} > dtau,

where phi_{s,t} is the flow of v from time s to t, i.e. phi_{s,t} = exp((t - s) v) for a stationary v. Its L^2 adjoint, applied to an
image f~, is (the change of variables y = phi_{1,tau}(x) gives the Jacobian |D phi_{tau,1}|)

    [dW|_f(v)]^* f~ = - int_0^1 |D phi_{tau,1}| (f~ o phi_{tau,1}) grad(f o phi_{tau,0}) dtau,                                     (*)

with phi_{tau,0} = exp(-tau v) and phi_{tau,1} = exp((1 - tau) v). The V-adjoint (the RKHS of the velocities) is K applied to it, with
K the kernel operator, e.g. a Gaussian smoothing (see smooth). With f~ = W(f, v) - f_target, (*) is the L^2 gradient of
1/2 ||W(f, v) - f_target||^2 with respect to v.

The adjoint is computed by one of the methods
    autograd    the exact adjoint of the discretized operator, by backpropagating <W(f, v), f~> to v. Exact, but differentiating it
                (to train the nets that get it) needs the second derivatives of the interpolation: grid_sample has them, the grid_pull
                backend of torch-interpol does not (the double backward fails).
    gauss<M>    (*) by M-point Gauss-Legendre quadrature in tau, e.g. gauss4. Converges to the exact adjoint (up to the discretization
                of the gradient and the Jacobian), and is used to verify (*).
    right       one-point rule tau = 1: - f~ grad W(f, v). The gradient of the already computed prediction, so nearly free. It is the
                exact gradient for a perturbation composed after the deformation, W(f, v) o (id - eps u), i.e. the force of
                diffeomorphic demons (Vercauteren et al. 2009) and of the right-trivialized (compositive) update.
    left        one-point rule tau = 0: - |D exp(v)| (f~ o exp(v)) grad f.
    midpoint    one-point rule tau = 1/2: - |D exp(v/2)| (f~ o exp(v/2)) grad W(f, v/2). Second order in the size of v, symmetric.
    trapezoid   the mean of left and right.

The velocity channels are (x, y, z), i.e. the spatial axes (W, H, D) in reverse storage order, as the velocities of the model, and the
velocities are in physical units (mm), see FlowDeformationOperator._normalize. The spatial gradient uses the grid spacing of the
operator's normalization, extent / (size - 1), and by default the derivative of the interpolant of the group action (B-splines for
grid_pull, see spatial_gradient), so that the result is comparable to the autograd adjoint.
"""
from __future__ import annotations

import math
from typing import Sequence

import interpol
import numpy as np
import torch
from torch.nn.functional import conv1d, pad

from operators.lddmm.deform import FlowDeformationOperator

METHODS = ("autograd", "right", "left", "midpoint", "trapezoid")  # and gauss<M>


def spacing(deform: FlowDeformationOperator, shape: Sequence[int]) -> tuple[float, ...]:
    """The grid spacing in mm along the storage axes ((D,) H, W) in the convention of the operator: the extent spans size - 1 steps."""
    extent = deform.extent if deform.extent is not None else tuple(float(s) for s in shape)
    return tuple(e / (s - 1) for e, s in zip(extent, shape))


def spatial_gradient(f: torch.Tensor, h: Sequence[float], method: str = "central", order: int | Sequence[int] = 3) -> torch.Tensor:
    """The spatial gradient of f (B, D, H, W) with spacings h (D, H, W), as (B, 3, D, H, W) with the channels in the order of the
    velocities (x, y, z) = (W, H, D).
        central  central differences (one-sided at the boundary): the derivative of the linear interpolant (grid_sample) up to O(h^2).
        spline   the derivative of the B-spline interpolant of the given order at the grid points (torch-interpol), i.e. that of the
                 grid_pull backend, which is what the autograd adjoint differentiates. On CT edges the two differ by tens of percent
                 (see test/bench_lddmm_adjoint.py), so the gradient should match the backend of the group action."""
    nd = f.dim() - 1
    if method == "spline":
        grid = torch.stack(torch.meshgrid(*[torch.arange(s, device=f.device, dtype=f.dtype) for s in f.shape[1:]], indexing="ij"), dim=-1)
        orders = [order] * nd if isinstance(order, int) else list(order)
        g = interpol.grid_grad(f.unsqueeze(1), grid.expand(f.shape[0], *grid.shape), interpolation=orders, bound="dct2",  # type: ignore
                               extrapolate=False, prefilter=max(orders) > 1)[:, 0]                      # (B, D, H, W, nd), d/d voxel index
        return torch.stack([g[..., i] / h[i] for i in reversed(range(nd))], dim=1)
    assert method == "central", f"Unknown gradient {method}, expected 'central' or 'spline'."
    grads = []
    for i in range(nd):
        ax = i + 1
        fp = torch.cat([f.narrow(ax, 1, f.shape[ax] - 1), f.narrow(ax, f.shape[ax] - 1, 1)], dim=ax)  # f[k+1], replicated at the end
        fm = torch.cat([f.narrow(ax, 0, 1), f.narrow(ax, 0, f.shape[ax] - 1)], dim=ax)                # f[k-1], replicated at the start
        n = torch.full((f.shape[ax],), 2.0, device=f.device, dtype=f.dtype)
        n[0] = n[-1] = 1.0
        grads.append((fp - fm) / (n.view(*([1] * ax), -1, *([1] * (nd - i - 1))) * h[i]))
    return torch.stack(grads[::-1], dim=1)


def default_gradient(deform: FlowDeformationOperator) -> str:
    """The spatial gradient matching the interpolation of the group action, see spatial_gradient."""
    return "spline" if deform.group_action.backend == "grid_pull" else "central"


def smooth(u: torch.Tensor, sigma_mm: float, h: Sequence[float]) -> torch.Tensor:
    """K u for a Gaussian kernel of standard deviation sigma_mm (separable, replicate padding), for u (B, C, D, H, W)."""
    if sigma_mm <= 0:
        return u
    B, C = u.shape[:2]
    out = u
    for i, hi in enumerate(h):
        s = sigma_mm / hi
        r = max(1, int(math.ceil(3 * s)))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=u.device, dtype=u.dtype) / s) ** 2)
        k = (k / k.sum()).view(1, 1, -1)
        ax = 2 + i
        x = out.movedim(ax, -1)
        sh = x.shape
        x = pad(x.reshape(-1, 1, sh[-1]), (r, r), mode="replicate")
        out = conv1d(x, k).reshape(sh).movedim(-1, ax)
    return out


def flow(deform: FlowDeformationOperator, v: torch.Tensor, s: float) -> torch.Tensor:
    """exp(-s v) in the normalized coordinates of grid_sample, for v (B, 3, D, H, W) in mm. W(f, s v) = f o flow(v, s)."""
    vn = deform._normalize(s * v.permute(0, 2, 3, 4, 1))
    integ = deform.velocity_integrator
    if integ.id is None:
        integ._init_shape(tuple(vn.shape[1:-1]))
    return integ(vn)


def warp(deform: FlowDeformationOperator, f: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """f o phi for f (B, D, H, W), with the group action of the operator."""
    return deform.group_action(f, phi).squeeze(1)


def _tau_term(deform: FlowDeformationOperator, f: torch.Tensor, v: torch.Tensor, f_tilde: torch.Tensor, tau: float,
              h: Sequence[float], w: torch.Tensor | None = None, gradient: str = "central") -> torch.Tensor:
    """The integrand of (*) at tau: |D phi_{tau,1}| (f~ o phi_{tau,1}) grad(f o phi_{tau,0}). w = W(f, v) can be passed for tau = 1."""
    if tau == 1.0:
        J_tau = w if w is not None else warp(deform, f, flow(deform, v, 1.0))
        return f_tilde.unsqueeze(1) * spatial_gradient(J_tau, h, gradient, deform.group_action.order)
    J_tau = f if tau == 0.0 else warp(deform, f, flow(deform, v, tau))      # f o phi_{tau,0} = f o exp(-tau v)
    phi_t1 = flow(deform, v, -(1.0 - tau))                                   # phi_{tau,1} = exp((1 - tau) v)
    jac = deform.group_action.jacobian_determinant(phi_t1)
    return (jac * warp(deform, f_tilde, phi_t1)).unsqueeze(1) * spatial_gradient(J_tau, h, gradient, deform.group_action.order)


def adjoint_derivative(deform: FlowDeformationOperator, f: torch.Tensor, v: torch.Tensor, f_tilde: torch.Tensor, method: str = "right",
                       w: torch.Tensor | None = None, sigma_mm: float = 0.0, create_graph: bool | None = None,
                       gradient: str | None = None) -> torch.Tensor:
    r"""[dW|_f(v)]^* f~ (see the module docstring) for an image f (B, D, H, W), a velocity field v (B, 3, D, H, W) in mm and an image
    f~ (B, D, H, W), as (B, 3, D, H, W). w = W(f, v) avoids recomputing it (method right and trapezoid). sigma_mm > 0 applies K, a
    Gaussian smoothing, giving the V-adjoint. gradient: the spatial gradient of the quadrature methods, by default that matching the
    backend of the group action (see spatial_gradient). create_graph (autograd only, by default whether any input requires grad): keep the graph
    of the adjoint, to backpropagate through it."""
    h = spacing(deform, f.shape[1:])
    gradient = gradient if gradient is not None else default_gradient(deform)
    if method == "autograd":
        if create_graph is None:
            create_graph = torch.is_grad_enabled() and any(x.requires_grad for x in (f, v, f_tilde))
        with torch.enable_grad():
            v_ = v if (create_graph and v.requires_grad) else v.detach().requires_grad_(True)
            if not create_graph:
                f, f_tilde = f.detach(), f_tilde.detach()
            Wv = warp(deform, f, flow(deform, v_, 1.0))
            (g,) = torch.autograd.grad((Wv * f_tilde).sum(), v_, create_graph=create_graph)
        out = g
    elif method == "right":
        out = -_tau_term(deform, f, v, f_tilde, 1.0, h, w, gradient)
    elif method == "left":
        out = -_tau_term(deform, f, v, f_tilde, 0.0, h, gradient=gradient)
    elif method == "midpoint":
        out = -_tau_term(deform, f, v, f_tilde, 0.5, h, gradient=gradient)
    elif method == "trapezoid":
        out = -0.5 * (_tau_term(deform, f, v, f_tilde, 0.0, h, gradient=gradient) + _tau_term(deform, f, v, f_tilde, 1.0, h, w, gradient))
    elif method.startswith("gauss"):
        M = int(method[5:])
        nodes, weights = np.polynomial.legendre.leggauss(M)
        out = -sum(0.5 * float(wk) * _tau_term(deform, f, v, f_tilde, 0.5 * (float(xk) + 1.0), h, gradient=gradient) for xk, wk in zip(nodes, weights))
    else:
        raise ValueError(f"Unknown method {method}, expected one of {METHODS} or gauss<M>.")
    return smooth(out, sigma_mm, h)
