"""
Checks that integrating the negated velocity inverts the deformation, as needed for the backward coupling terms of model/hlpd_loss.py.

The integrator returns the sampling grid G = exp(-v) = phi^{-1} of W(f, v) = f o phi^{-1}, with phi = exp(v) the motion. For a stationary
velocity field exp(-v) = exp(v)^{-1}, so integrating -v gives H = exp(v) = phi, the grid that pulls an image back by the motion. Numerically,
H o G = G o H = id holds up to the interpolation error of the integrator, which is checked here on a smooth swirl (with a shear in 3D) of a
few voxels, vanishing at the boundary so that the border padding of _compose does not enter (see test/test_velocity_integration.py).

The errors are in voxels, relative to the largest displacement of the motion: with N = 7, about 1% (grid_pull) to 2% (grid_sample).

Run: pytest test/test_inverse_consistency.py
Plot: python -m test.test_inverse_consistency (writes test/plots/inverse_consistency_grid.png)
"""
import math

import matplotlib.pyplot as plt
import pytest
import torch

from operators.lddmm.deform import VelocityIntegrator
from test.plotting import PLOT_DIR, plot_deformed_grid

N = 7  # the default of model.hlpd.DeformParams
BACKENDS = [("grid_sample", 1), ("grid_pull", 3)]


def _voxel_size(shape: tuple[int, ...]) -> torch.Tensor:
    """The size of a voxel in grid_sample's normalized units, channels (x, y[, z])."""
    return torch.tensor([2 / (s - 1) for s in reversed(shape)])


def _swirl(shape: tuple[int, ...], amplitude: float) -> torch.Tensor:
    """A smooth (1, *shape, C) velocity in normalized units whose largest displacement is `amplitude` voxels, zero at the boundary."""
    grid = VelocityIntegrator(N=1, shape=shape).id.unsqueeze(0)  # type: ignore[union-attr]
    bump = torch.prod(torch.cos(math.pi / 2 * grid) ** 2, dim=-1)
    v = torch.zeros_like(grid)
    v[..., 0], v[..., 1] = -grid[..., 1] * bump, grid[..., 0] * bump
    if len(shape) == 3:
        v[..., 2] = 0.5 * torch.sin(math.pi * grid[..., 0]) * bump
    return v / (v / _voxel_size(shape)).abs().amax() * amplitude


def _error(phi: torch.Tensor, vi: VelocityIntegrator, shape: tuple[int, ...]) -> torch.Tensor:
    """The distance of phi from the identity in voxels, (1, *shape)."""
    return ((phi - vi.id) / _voxel_size(shape)).norm(dim=-1)


@pytest.mark.parametrize("backend,order", BACKENDS)
@pytest.mark.parametrize("shape", [(64, 64), (24, 48, 48)])
@pytest.mark.parametrize("amplitude", [1.0, 3.0, 6.0])
def test_negated_velocity_inverts_the_deformation(backend: str, order: int, shape: tuple[int, ...], amplitude: float):
    vi = VelocityIntegrator(N=N, shape=shape, backend=backend, order=order)
    v = _swirl(shape, amplitude)

    G, H = vi(v), vi(-v)

    assert _error(G, vi, shape).max() > 0.5 * amplitude  # the motion itself is far from the identity
    for name, composed in {"H o G": vi._compose(H, G), "G o H": vi._compose(G, H)}.items():
        err = _error(composed, vi, shape)
        assert err.max() < 0.025 * amplitude, f"{name}: max error {err.max():.4f} voxels for a {amplitude:g} voxel motion"
        assert err.mean() < 0.005 * amplitude, f"{name}: mean error {err.mean():.5f} voxels for a {amplitude:g} voxel motion"


def plot_inverse_consistency(shape: tuple[int, int] = (64, 64), amplitude: float = 6.0, backend: str = "grid_sample", order: int = 1,
                             savepath=PLOT_DIR / "inverse_consistency_grid.png"):
    """Plots phi = exp(v), phi^{-1} = exp(-v), phi o phi^{-1} and the distance of the latter from the identity."""
    vi = VelocityIntegrator(N=N, shape=shape, backend=backend, order=order)
    v = _swirl(shape, amplitude)
    G, H = vi(v), vi(-v)
    composed = vi._compose(H, G)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plot_deformed_grid(H, ax=axes[0], title=r"$\phi = \exp(v)$, integrator($-v$)")
    plot_deformed_grid(G, ax=axes[1], title=r"$\phi^{-1} = \exp(-v)$, integrator($v$)")
    plot_deformed_grid(composed, ax=axes[2], title=r"$\phi \circ \phi^{-1}$")
    err = _error(composed, vi, shape)[0]
    im = axes[3].imshow(err, cmap="magma")
    axes[3].set_title(rf"$|\phi \circ \phi^{{-1}} - \mathrm{{id}}|$ [voxels], max {err.max():.3f}")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    fig.suptitle(f"{amplitude:g} voxel swirl, N = {N}, {backend} (order {order})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    savepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving inverse consistency plot to: {savepath}")
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


if __name__ == "__main__":
    for backend, order in BACKENDS:
        plot_inverse_consistency(backend=backend, order=order, savepath=PLOT_DIR / f"inverse_consistency_grid_{backend}.png")
