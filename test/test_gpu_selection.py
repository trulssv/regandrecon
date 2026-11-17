"""
Checks the GPU selection of the training scripts (model.gpus.free_gpus / select_gpu) against mocked nvidia-smi output: the GPUs are
matched to torch's indices by UUID, busy GPUs are skipped, and ranks take the free GPUs in index order.

Run: pytest test/test_gpu_selection.py
"""
import subprocess
from types import SimpleNamespace

import pytest

import model.gpus as build

UUIDS = ["aaaa", "bbbb", "cccc", "dddd"]


@pytest.fixture
def four_gpus(monkeypatch):
    """Four GPUs, which nvidia-smi lists in a different order than torch: GPU 0 (torch) is busy, GPU 2 is idle but holds 3 GB."""
    smi = {"aaaa": (24780, 99), "bbbb": (14, 0), "cccc": (3000, 0), "dddd": (68, 0)}
    lines = [f"GPU-{uuid}, {used}, {util}" for uuid, (used, util) in reversed(smi.items())]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="\n".join(lines) + "\n"))
    monkeypatch.setattr(build.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(build.torch.cuda, "get_device_properties", lambda i: SimpleNamespace(uuid=UUIDS[i]))
    selected = {}
    monkeypatch.setattr(build.torch.cuda, "set_device", lambda i: selected.setdefault("torch", i))
    import astra
    monkeypatch.setattr(astra, "set_gpu_index", lambda i: selected.setdefault("astra", i))
    return selected


def test_free_gpus_skips_busy_gpus_and_matches_by_uuid(four_gpus):
    assert build.gpu_usage()[0] == pytest.approx((24780 * 2**20 / 1e9, 99.0))
    assert build.free_gpus() == [1, 3]
    assert build.free_gpus(max_used_gb=4.0) == [1, 2, 3]


def test_ranks_take_the_free_gpus_in_order(four_gpus):
    device = build.select_gpu(local_rank=1, local_world_size=2)
    assert device.index == 3 and four_gpus == {"torch": 3, "astra": 3}


def test_too_few_free_gpus_is_an_error(four_gpus):
    with pytest.raises(RuntimeError, match="Only 2 free GPUs"):
        build.select_gpu(local_rank=0, local_world_size=3)


def test_explicit_gpus(four_gpus):
    assert build.select_gpu(local_rank=1, local_world_size=2, gpus="2,0").index == 0
