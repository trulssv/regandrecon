"""
Checks of operators.lddmm.helmholtz.HelmholtzOperator, L = (alpha - gamma Laplacian)^beta with physical (mm) coordinates from the extent:
the output has the shape of the input, L is the identity for gamma = 0, and a plane wave of frequency k (cycles/mm) is an eigenfunction
with eigenvalue (alpha + gamma 4 pi^2 |k|^2)^beta.

Run: pytest test/test_helmholtz.py
"""
import math

import pytest
import torch

from operators.lddmm.helmholtz import HelmholtzOperator

CASES = [((8, 10, 12), (16.0, 20.0, 30.0)), ((7, 9, 11), (14.0, 9.0, 33.0)), ((10, 12), (20.0, 30.0)), ((9, 11), (9.0, 22.0))]


@pytest.mark.parametrize("shape, extent", CASES)
def test_shape_is_preserved(shape, extent):
    op = HelmholtzOperator(dim=len(shape), extent=extent)
    v = torch.randn(2, *shape, len(shape))
    assert op(v).shape == v.shape


@pytest.mark.parametrize("shape, extent", CASES)
def test_identity_without_laplacian(shape, extent):
    op = HelmholtzOperator(dim=len(shape), alpha=1.0, gamma=0.0, beta=1.0, extent=extent)
    v = torch.randn(2, *shape, len(shape), dtype=torch.float64)
    torch.testing.assert_close(op(v), v)


@pytest.mark.parametrize("shape, extent", CASES)
@pytest.mark.parametrize("alpha, gamma, beta", [(1.0, 1.0, 1.0), (0.5, 4.0, 2.0)])
def test_plane_wave_is_an_eigenfunction(shape, extent, alpha, gamma, beta):
    """v_c = cos(2 pi k . x) with x the physical voxel positions, a different k per component and batch element."""
    dim = len(shape)
    spacing = [e / s for e, s in zip(extent, shape)]
    coords = torch.meshgrid(*[torch.arange(s, dtype=torch.float64) * h for s, h in zip(shape, spacing)], indexing="ij")
    op = HelmholtzOperator(dim=dim, alpha=alpha, gamma=gamma, beta=beta, extent=extent)

    v = torch.zeros(2, *shape, dim, dtype=torch.float64)
    expected = torch.zeros_like(v)
    for b in range(2):
        for c in range(dim):
            n = [(b + c + i) % 3 + (i == c) for i in range(dim)]               # integer number of periods along each axis
            k = [ni / (s * h) for ni, s, h in zip(n, shape, spacing)]          # cycles/mm
            wave = torch.cos(2 * math.pi * sum(ki * xi for ki, xi in zip(k, coords)))
            v[b, ..., c] = wave
            expected[b, ..., c] = (alpha + gamma * 4 * math.pi**2 * sum(ki**2 for ki in k)) ** beta * wave

    torch.testing.assert_close(op(v), expected, rtol=1e-5, atol=1e-5 * expected.abs().max().item())  # the frequency grid is float32
