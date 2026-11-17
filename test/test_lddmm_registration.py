"""
Tests of the test-phase registration modules (model/motion.py): the U-net, and HLPD models whose motion nets are replaced by
U-nets and/or the LDDMM-styled motion update, against the mock ray transform of test/test_hlpd.py.
"""
import pytest
import torch

from model.hlpd import ConvStack, ConvStackParams
from model.motion import LDDMMBlock, LDDMMParams, UNet3D, UNetParams, convert_motion
from test.test_hlpd import B, SHAPE, SINO_SHAPE, T, make_hlpd_model


def test_unet_shapes_and_zero_init():
    net = UNet3D(9, 3, **UNetParams().__dict__)
    x = torch.randn(2, 9, 25, 30, 34)  # odd sizes
    y = net(x)
    assert y.shape == (2, 3, 25, 30, 34)
    assert net.outputs_zero() and not y.any()
    torch.nn.init.normal_(net.final.weight)
    assert not net.outputs_zero()


VARIANTS = [
    pytest.param("conv", None, id="conv"),
    pytest.param("unet", None, id="unet"),
    pytest.param("conv", LDDMMParams(), id="conv-lddmm-right-residual"),
    pytest.param("unet", LDDMMParams(method="midpoint", adjoint_input="target", sigma_mm=2.0), id="unet-lddmm-midpoint-target"),
]


def randomize_final(model):
    """Nonzero final layers, so that the motion is not zero and the gradients reach all layers."""
    torch.manual_seed(3)
    for block in model.unrolled_model.hlpd_model_list:
        for net in block.motion_net:
            final = net.final if isinstance(net, UNet3D) else net.blocks[-1].conv
            torch.nn.init.normal_(final.weight, std=1e-2)


@pytest.mark.parametrize("kind,lddmm", VARIANTS)
def test_converted_model_trains(monkeypatch, kind, lddmm):
    model = make_hlpd_model(monkeypatch, recurrent=True)
    primal_before = {k: v.clone() for k, v in model.state_dict().items() if "primal_net" in k or "dual_net" in k}
    convert_motion(model, kind, lddmm)
    # The primal and dual nets are kept.
    after = model.state_dict()
    assert all(torch.equal(after[k], v) for k, v in primal_before.items())
    blocks = list(model.unrolled_model.hlpd_model_list)
    assert all(isinstance(b, LDDMMBlock) == (lddmm is not None) for b in blocks)
    assert all(isinstance(n, UNet3D if kind == "unet" else ConvStack) for b in blocks for n in b.motion_net)

    randomize_final(model)
    model.set_trainable(["motion"])
    assert not model.skip_motion
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    w, f, v = model(g)
    assert all(vt.shape == (B, *SHAPE, 3) for vt in v) and all(vt.abs().sum() > 0 for vt in v)
    loss = sum((wt - ft).square().sum() for wt, ft in zip(w, f))
    loss.backward()
    motion = [(n, p) for n, p in model.named_parameters() if "motion_net" in n]
    assert all(p.grad is not None for _, p in motion)
    assert all(p.grad is None for n, p in model.named_parameters() if "motion_net" not in n)


@pytest.mark.parametrize("kind,lddmm", VARIANTS[2:])
def test_lddmm_checkpointing_matches_plain(monkeypatch, kind, lddmm):
    outs, grads = [], []
    for ckpt in (False, True):
        model = make_hlpd_model(monkeypatch, recurrent=True)
        convert_motion(model, kind, lddmm)
        randomize_final(model)
        model.unrolled_model.grad_checkpointing = ckpt
        torch.manual_seed(5)
        g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
        w, f, v = model(g)
        sum(wt.square().sum() for wt in w).backward()
        outs.append(torch.stack(w))
        grads.append(torch.cat([p.grad.flatten() for n, p in model.named_parameters() if "motion_net" in n and p.grad is not None]))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    assert torch.allclose(grads[0], grads[1], rtol=1e-3, atol=1e-6)


def test_untrained_unet_skips_motion(monkeypatch):
    model = make_hlpd_model(monkeypatch, recurrent=True)
    convert_motion(model, "unet", LDDMMParams())
    model.set_trainable(["dual", "primal"])
    assert model.skip_motion
    g = [torch.randn(B, *SINO_SHAPE) for _ in range(T)]
    w, f, v = model(g)
    assert all(not vt.any() for vt in v)


# The motion section of the model config (MotionParams), and resuming a checkpoint with other motion nets (train.train.load_model_weights).

import json
from dataclasses import asdict
from pathlib import Path

from model.build import HLPDConfig
from model.motion import MotionParams, apply_motion_config
from train.train import load_model_weights

LDDMM_CONFIG = Path("train/configs/hlpd_staged_unet_lddmm.json")


def test_motion_config_round_trip_and_default():
    assert MotionParams().is_default and HLPDConfig.from_dict({}).motion.is_default
    cfg = json.loads(LDDMM_CONFIG.read_text())["model"]
    config = HLPDConfig.from_dict(cfg)
    assert config.motion.net == "unet" and config.motion.lddmm is not None and config.motion.lddmm.adjoint_scale == 86.0
    # the run config (HLPDConfig.to_dict, stored in the checkpoints) restores the same config, also without a motion section (old runs)
    assert HLPDConfig.from_dict(config.to_dict()).motion == config.motion
    assert HLPDConfig.from_dict({**cfg, "motion": None}).motion.is_default
    with pytest.raises(AssertionError, match="Unknown keys"):
        MotionParams.from_dict({"nets": "unet"})
    with pytest.raises(AssertionError, match="Unknown motion net"):
        MotionParams.from_dict({"net": "resnet"})


def build(monkeypatch, motion: MotionParams):
    model = make_hlpd_model(monkeypatch, recurrent=True)
    return apply_motion_config(model, motion, ConvStackParams(hidden_channels=[4]))


def checkpoint_of(model, motion: MotionParams) -> dict:
    return {"model": {k: v.clone() for k, v in model.state_dict().items()}, "config": {"motion": asdict(motion)}}


def test_apply_motion_config(monkeypatch):
    blocks = lambda m: list(m.unrolled_model.hlpd_model_list)
    plain = build(monkeypatch, MotionParams())
    assert all(type(b).__name__ == "HLPDBlock" and isinstance(b.motion_net[0], ConvStack) for b in blocks(plain))
    model = build(monkeypatch, MotionParams(net="unet", lddmm=LDDMMParams()))
    assert all(isinstance(b, LDDMMBlock) and isinstance(b.motion_net[0], UNet3D) and b.motion_net[0].outputs_zero() for b in blocks(model))
    # the exact adjoint cannot be trained through with the grid_pull backend (no double backward)
    pulled = make_hlpd_model(monkeypatch, recurrent=True)
    pulled.deform.group_action.backend = "grid_pull"
    with pytest.raises(AssertionError, match="grid_pull"):
        apply_motion_config(pulled, MotionParams(lddmm=LDDMMParams(method="autograd")), ConvStackParams())


def test_resume_with_new_motion_nets(monkeypatch):
    old_motion, new_motion = MotionParams(), MotionParams(net="unet", lddmm=LDDMMParams())
    old = build(monkeypatch, old_motion)
    randomize_final(old)  # trained motion nets, which are discarded
    ckpt = checkpoint_of(old, old_motion)

    new = build(monkeypatch, new_motion)
    with pytest.raises(AssertionError, match="--start-stage"):
        load_model_weights(new, ckpt, new_motion, new_motion_ok=False)
    load_model_weights(new, ckpt, new_motion, new_motion_ok=True)
    state = new.state_dict()
    assert all(torch.equal(state[k], v) for k, v in ckpt["model"].items() if ".motion_net." not in k)
    assert all(n.outputs_zero() for b in new.unrolled_model.hlpd_model_list for n in b.motion_net)

    # the same motion config loads everything, the motion nets included
    again = build(monkeypatch, new_motion)
    randomize_final(new)
    load_model_weights(again, checkpoint_of(new, new_motion), new_motion, new_motion_ok=False)
    assert all(torch.equal(again.state_dict()[k], v) for k, v in new.state_dict().items())
