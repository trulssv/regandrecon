"""
Checks that both VelocityIntegrator schemes integrate the full velocity field, on a constant velocity (a translation) whose flow is known
in closed form: phi = id - v.

Only the interior is checked: _compose resamples with padding_mode="border", which clamps the field wherever a translation samples outside
the domain (see research/lddmm/BOUNDARY_FLUX_STATUS_REPORT.md).

This guards against scale_and_square squaring N-1 instead of N times after scaling v by 1/2^N, which integrated v/2.

Run: pytest test/test_velocity_integration.py
"""
import pytest
import torch

from operators.lddmm.deform import VelocityIntegrator

SHAPE = (16, 16, 16)
MARGIN = 4  # voxels excluded at each edge, more than the translation below


@pytest.mark.parametrize("integration", ["euler", "scaling_and_squaring"])
@pytest.mark.parametrize("N", [1, 3, 5])
def test_translation_is_integrated_fully(integration: str, N: int):
    vi = VelocityIntegrator(N=N, integration=integration, shape=SHAPE)
    assert vi.id is not None
    v = torch.zeros(1, *SHAPE, 3)
    v[..., 0], v[..., 1], v[..., 2] = 0.1, 0.0, -0.05  # normalized units, about 0.8 and 0.4 voxels

    phi = vi(v)

    interior = (slice(None),) + (slice(MARGIN, -MARGIN),) * 3
    torch.testing.assert_close(phi[interior], (vi.id - v)[interior], atol=1e-5, rtol=0)


@pytest.mark.parametrize("integration", ["euler", "scaling_and_squaring"])
def test_superres_ends_at_the_full_flow(integration: str):
    N = 4
    vi = VelocityIntegrator(N=N, integration=integration, shape=SHAPE)
    v = torch.zeros(1, *SHAPE, 3)
    v[..., 0] = 0.1

    phi_list = vi(v, superres=True)

    assert len(phi_list) == N
    torch.testing.assert_close(phi_list[-1], vi(v), atol=1e-6, rtol=0)
