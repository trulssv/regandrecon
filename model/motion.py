r"""
Registration (motion) modules of the HLPD model, selected by the "motion" section of the model config (MotionParams, applied by
build_hlpd in model/build.py), and compared in test/compare_registration_modules.py:

UNet3D
    A small 3D U-net as the motion net, for a larger field of view than the ConvStack (whose 4 convs see 9 voxels, ~16 mm in-plane). The
    first level only downsamples in-plane, as the volumes are anisotropic (3 mm slices, 1.76 mm pixels). Same interface as ConvStack
    (zero initialized final conv, outputs_zero), so that it works with HLPDModel.set_trainable and the residual updates.

LDDMMBlock
    The recurrent HLPD block with LDDMM-styled motion modules,
        v_t^{(i)} = v_t^{(i-1)} + Sigma(v_t^{(i-1)}, v_{t-1}^{(i)}, a * [dW|_{f_t^{(i)}}(v_t^{(i-1)})]^* r_t),
    where the adjoint derivative (operators/lddmm/adjoint.py) is evaluated at the previous iterate of the velocity being updated, with the
    current template f_t, and applied to
        residual  r_t = W(f_t, v_t^{(i-1)}) - f_{t+1}: then it is the gradient of 1/2 ||W(f_t, v) - f_{t+1}||^2, the analogue of the
                  T^* h input of the primal nets (the default)
        target    r_t = f_{t+1}, as in the module proposal
    and a is a fixed scale (adjoint_scale) putting it on the order of the velocities in the first conv. The images enter the motion net
    only through the adjoint. The primal and dual updates are those of HLPDBlock.

convert_motion
    Replaces the motion nets of an HLPDModel (e.g. one with trained primal and dual nets) by new ones (ConvStack or UNet3D, with the
    inputs of HLPDBlock or of LDDMMBlock), keeping everything else.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Tuple, cast

import torch
from torch import nn
from torch.nn import functional as F

from model.hlpd import ConvBlock, ConvStack, ConvStackParams, HLPDBlock, HLPDModel, net_channels
from operators.lddmm.adjoint import adjoint_derivative
from tracking.timeline import region


@dataclass
class UNetParams:
    channels: list[int] = field(default_factory=lambda: [16, 32, 64])          # per level, the last one is the bottleneck
    down: list[list[int]] = field(default_factory=lambda: [[1, 2, 2], [2, 2, 2]])  # downsampling factors (D, H, W) between the levels
    convs_per_level: int = 2
    batch_norm: bool = True     # GroupNorm(4), as ConvStack
    zero_init_final: bool = True


class UNet3D(nn.Module):
    """U-net with average pooling down, trilinear upsampling to the size of the skip connection (so odd sizes, e.g. 25 slices, work),
    and a final 3^3 conv without activation."""

    def __init__(self, in_channels: int, out_channels: int, channels: list[int], down: list[list[int]], convs_per_level: int = 2,
                 batch_norm: bool = True, zero_init_final: bool = True) -> None:
        super().__init__()
        assert len(down) == len(channels) - 1, f"Expected {len(channels) - 1} downsampling factors for {len(channels)} levels, got {down}."
        self.down = [tuple(d) for d in down]

        def level(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(*[ConvBlock(c_in if k == 0 else c_out, c_out, batch_norm=batch_norm) for k in range(convs_per_level)])

        self.enc = nn.ModuleList()
        c_prev = in_channels
        for c in channels:
            self.enc.append(level(c_prev, c))
            c_prev = c
        self.dec = nn.ModuleList([level(channels[k + 1] + channels[k], channels[k]) for k in reversed(range(len(channels) - 1))])
        self.final = nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=1)
        if zero_init_final:
            nn.init.zeros_(self.final.weight)
            assert self.final.bias is not None
            nn.init.zeros_(self.final.bias)

    def outputs_zero(self) -> bool:
        return not self.final.weight.any() and (self.final.bias is None or not self.final.bias.any())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for k, enc in enumerate(self.enc):
            if k > 0:
                x = F.avg_pool3d(x, kernel_size=self.down[k - 1], stride=self.down[k - 1], ceil_mode=True)
            x = enc(x)
            skips.append(x)
        for dec, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.final(x)


@dataclass
class LDDMMParams:
    method: str = "right"           # approximation of the adjoint, see operators/lddmm/adjoint.py
    adjoint_input: str = "residual"  # residual or target, see the module docstring
    adjoint_scale: float = 10.0     # a: the adjoint is ~ (image units)^2 / mm, i.e. ~1e-2 for images in units of water and mm velocities
    sigma_mm: float = 0.0           # > 0: the V-adjoint, with a Gaussian K of this width

    def __post_init__(self) -> None:
        assert self.adjoint_input in ("residual", "target"), self.adjoint_input


class LDDMMBlock(HLPDBlock):
    """HLPDBlock (recurrent) with the LDDMM-styled motion update, see the module docstring. Built from an HLPDBlock by convert_motion."""

    lddmm: LDDMMParams

    def _motion_input(self, f_t: torch.Tensor, v_prev: torch.Tensor, v_t: torch.Tensor, f_target: torch.Tensor) -> torch.Tensor:
        p = self.lddmm
        with region("hlpd/lddmm_adjoint"):
            w = self._warp(f_t, v_prev) if (p.method in ("right", "trapezoid") or p.adjoint_input == "residual") else None
            f_tilde = (w - f_target) if p.adjoint_input == "residual" else f_target   # type: ignore[operator]
            adj = adjoint_derivative(self.deform, f_t, v_prev, f_tilde, p.method, w=w, sigma_mm=p.sigma_mm)
        return torch.cat([v_prev, v_t, p.adjoint_scale * adj], dim=1)

    def _time_step(self, t: int, f_prev: torch.Tensor, h_prev: torch.Tensor, v_prev: torch.Tensor, w_t: torch.Tensor, f_t: torch.Tensor,
                   h_t: torch.Tensor, v_t: torch.Tensor, f_target: torch.Tensor, w_target: torch.Tensor, g_t: torch.Tensor,
                   zero_primal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: the reconstruction part is HLPDBlock's, with the motion skipped, and the motion update follows on its outputs.
        skip = self.skip_motion
        self.skip_motion = True
        try:
            f_t, h_t, _, _ = super()._time_step(t, f_prev, h_prev, v_prev, w_t, f_t, h_t, v_t, f_target, w_target, g_t, zero_primal)
        finally:
            self.skip_motion = skip
        if skip:
            return f_t, h_t, v_prev, f_t
        assert isinstance(self.motion_net, nn.ModuleList)
        sigma_module = self.motion_net[0 if self.shared else t]
        with region("hlpd/motion_net"):
            v_t = v_prev + sigma_module(self._motion_input(f_t, v_prev, v_t, f_target))
        return f_t, h_t, v_t, self._warp(f_t, v_t)


@dataclass
class MotionParams:
    """The "motion" section of the model config: the kind of motion net ("conv": the ConvStack of the conv_stack section, as before;
    "unet": UNet3D with the unet section) and, if lddmm is given, the LDDMM-styled motion update (LDDMMBlock). The default, conv without
    lddmm, is the original HLPD model, so configs and checkpoints without this section are unchanged."""
    net: str = "conv"
    unet: UNetParams = field(default_factory=UNetParams)
    lddmm: LDDMMParams | None = None

    def __post_init__(self) -> None:
        assert self.net in ("conv", "unet"), f"Unknown motion net {self.net}, expected 'conv' or 'unet'."

    @classmethod
    def from_dict(cls, config: dict) -> "MotionParams":
        unknown = set(config) - {"net", "unet", "lddmm"}
        assert not unknown, f"Unknown keys {sorted(unknown)} in the motion config {config}."
        lddmm = config.get("lddmm")
        return cls(net=config.get("net", "conv"), unet=UNetParams(**config.get("unet", {})),
                   lddmm=LDDMMParams(**lddmm) if lddmm is not None else None)

    @property
    def is_default(self) -> bool:
        return self.net == "conv" and self.lddmm is None


def apply_motion_config(model: HLPDModel, motion: MotionParams, conv_stack: ConvStackParams) -> HLPDModel:
    """Replaces the motion nets of a freshly built HLPDModel according to the motion config (nothing for the default). Does not reseed
    the global RNG (the training seeds it once before building the model, and DDP broadcasts rank 0's parameters)."""
    if motion.is_default:
        return model
    assert model.block_params.recurrent, "The motion config (net 'unet' or lddmm) requires the recurrent model (block.recurrent)."
    if motion.lddmm is not None and motion.lddmm.method == "autograd":
        assert model.deform.group_action.backend == "grid_sample", \
            "The lddmm method 'autograd' cannot be trained with the grid_pull backend (no double backward), use e.g. 'right'."
    return convert_motion(model, motion.net, motion.lddmm, conv_stack=conv_stack, unet=motion.unet, seed=None)


def make_motion_net(kind: str, in_channels: int, out_channels: int, conv_stack: ConvStackParams | None = None,
                    unet: UNetParams | None = None) -> nn.Module:
    if kind == "conv":
        params = conv_stack if conv_stack is not None else ConvStackParams(hidden_channels=[32, 32, 32])
        return ConvStack(in_channels, out_channels, **{**params.__dict__})
    if kind == "unet":
        params = unet if unet is not None else UNetParams()
        return UNet3D(in_channels, out_channels, **params.__dict__)
    raise ValueError(f"Unknown motion net {kind}, expected 'conv' or 'unet'.")


def convert_motion(model: HLPDModel, kind: str = "unet", lddmm: LDDMMParams | None = None, conv_stack: ConvStackParams | None = None,
                   unet: UNetParams | None = None, seed: int | None = 0) -> HLPDModel:
    """Replaces the motion nets of every block of the (recurrent) model in place by new, zero initialized ones of the given kind, with
    the LDDMM-styled inputs if lddmm is given (the block then becomes an LDDMMBlock) and the inputs of HLPDBlock otherwise. The primal
    and dual nets are kept. Call model.set_trainable afterwards, as the motion nets change."""
    assert model.block_params.recurrent, "Only the recurrent model is supported."
    if seed is not None:
        torch.manual_seed(seed)
    _, _, _, _, m_in, m_out = net_channels(True, model.T)
    if lddmm is not None:
        m_in = 9    # v_prev, v_{t-1}, adjoint
    device = next(model.parameters()).device
    for block in model.unrolled_model.hlpd_model_list:
        block = cast(HLPDBlock, block)
        n = len(block.motion_net)  # type: ignore[arg-type]
        template = make_motion_net(kind, m_in, m_out, conv_stack, unet)
        block.motion_net = nn.ModuleList([copy.deepcopy(template) for _ in range(n)]).to(device)
        if lddmm is not None:
            block.__class__ = LDDMMBlock
            cast(LDDMMBlock, block).lddmm = lddmm
        elif isinstance(block, LDDMMBlock):
            block.__class__ = HLPDBlock
    return model


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
