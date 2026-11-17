"""The helpers of the time bin perturbation tests (test/time_bin_perturbation.py): rolling the time bins, substituting the nets of the
time bins (and restoring them), weight averages and distances, and the 3D SSIM. Runs on the CPU without data.

Run: pytest test/test_time_bin_perturbation.py
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test.time_bin_perturbation import (NET_GROUPS, assigned_nets, average_net, averaged_nets, roll_bins, rolled_nets, single_net, ssim3d,
                                        unique_blocks, weight_distances)

T, K = 4, 3


class FakeBlock(nn.Module):
    """The attributes of model.hlpd.HLPDBlock that the substitution uses."""
    def __init__(self) -> None:
        super().__init__()
        self.shared = False
        self.time_steps = T
        for group in NET_GROUPS:
            setattr(self, f"{group}_net", nn.ModuleList([nn.Linear(2, 2) for _ in range(T)]))


def fake_model(shared_iterations: bool = False) -> SimpleNamespace:
    blocks = [FakeBlock()] * K if shared_iterations else [FakeBlock() for _ in range(K)]
    return SimpleNamespace(unrolled_model=SimpleNamespace(hlpd_model_list=nn.ModuleList(blocks)))


def nets_of(model, group: str) -> list[list[nn.Module]]:
    return [list(getattr(block, f"{group}_net")) for block in unique_blocks(model)]


def test_roll_bins() -> None:
    assert roll_bins([0, 1, 2, 3], 1) == [1, 2, 3, 0]
    assert roll_bins([0, 1, 2, 3], -1) == [3, 0, 1, 2]
    assert roll_bins([0, 1, 2, 3], 4) == [0, 1, 2, 3]


def test_shared_iterations_are_substituted_once() -> None:
    model = fake_model(shared_iterations=True)
    assert len(unique_blocks(model)) == 1
    trained = nets_of(model, "dual")[0]
    with assigned_nets(model, rolled_nets(1)):
        assert nets_of(model, "dual")[0] == roll_bins(trained, 1)  # rolled once, not K times


@pytest.mark.parametrize("assignment, expected", [(rolled_nets(1), lambda nets: roll_bins(nets, 1)), (single_net(2), lambda nets: [nets[2]] * T)])
def test_assigned_nets_are_restored(assignment, expected) -> None:
    model = fake_model()
    trained = {group: nets_of(model, group) for group in NET_GROUPS}
    with assigned_nets(model, assignment, groups=("motion",)):
        assert nets_of(model, "motion") == [expected(nets) for nets in trained["motion"]]
        assert nets_of(model, "dual") == trained["dual"] and nets_of(model, "primal") == trained["primal"]
    assert {group: nets_of(model, group) for group in NET_GROUPS} == trained


def test_assigned_nets_are_restored_after_an_error() -> None:
    model = fake_model()
    trained = nets_of(model, "primal")
    with pytest.raises(RuntimeError):
        with assigned_nets(model, single_net(0)):
            raise RuntimeError
    assert nets_of(model, "primal") == trained


def test_average_net() -> None:
    nets = [nn.Linear(2, 2) for _ in range(T)]
    avg = average_net(nets)
    assert torch.allclose(avg.weight, torch.stack([n.weight for n in nets]).mean(0))
    assert all(avg is not n for n in nets)
    model = fake_model()
    with assigned_nets(model, averaged_nets(), groups=("dual",)):
        nets = nets_of(model, "dual")[0]
        assert all(n is nets[0] for n in nets)


def test_weight_distances() -> None:
    a, b = nn.Linear(2, 2), nn.Linear(2, 2)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        b.weight.mul_(-1.0)
        b.bias.mul_(-1.0)
    d = weight_distances([a, a, b])
    assert d.shape == (3, 3)
    assert d[0, 1] == pytest.approx(0.0, abs=1e-6) and d[0, 2] == pytest.approx(2.0, rel=1e-5)  # ||2 theta|| / ||theta||


def test_ssim3d() -> None:
    torch.manual_seed(0)
    x = torch.rand(1, 12, 12, 12)
    assert ssim3d(x, x) == pytest.approx(1.0)
    small, large = ssim3d(x + 0.05 * torch.randn_like(x), x), ssim3d(x + 0.3 * torch.randn_like(x), x)
    assert 1.0 > small > large
