"""
Builds the HLPD model and its loss (model/hlpd_loss.py) from a config, shared by the training scripts (train/train.py, test/train_hlpd_overfit.py).

The config holds the architecture and loss parameters, as dicts of the dataclasses in model/hlpd.py and of LossParams in model/hlpd_loss.py
(one section per loss term, see there):
    {"unrolled": {...}, "block": {...}, "conv_stack": {...}, "deform": {...}, "normalization": {...}, "motion": {...}, "loss": {...}, "zero_init": bool}
The motion section (MotionParams in model/motion.py) selects the motion nets: the ConvStack (the default), a U-net, and/or the LDDMM-styled
motion update, e.g. {"net": "unet", "unet": {"channels": [16, 32, 64]}, "lddmm": {"method": "right", "adjoint_scale": 86.0}}.
The ray transform is not part of it: it must match the geometry the data was simulated with, see simulation_ray_trafo_params.
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from model.gpus import configure_cuda, free_gpus, gpu_usage, select_gpu  # noqa: F401, re-exported for the training scripts
from model.hlpd import ConvStackParams, DeformParams, HLPDBlockParams, HLPDModel, HLPDUnrolledParams, NormalizationParams, RayTrafoParams
from model.motion import MotionParams, apply_motion_config
from model.hlpd_loss import HLPDLoss, LossParams  # noqa: F401, LossParams re-exported for the training scripts
from operators.lddmm.helmholtz import HelmholtzOperator

logger = logging.getLogger(__name__)


@dataclass
class HLPDConfig:
    unrolled: HLPDUnrolledParams = field(default_factory=HLPDUnrolledParams)
    block: HLPDBlockParams = field(default_factory=HLPDBlockParams)
    conv_stack: ConvStackParams = field(default_factory=ConvStackParams)
    deform: DeformParams = field(default_factory=DeformParams)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    motion: MotionParams = field(default_factory=MotionParams)
    loss: LossParams = field(default_factory=LossParams)
    zero_init: bool = True                                              # zero initialization of (f, h), otherwise f is initialized with the FBP and h with the data

    @classmethod
    def from_dict(cls, config: dict) -> "HLPDConfig":
        return cls(
            unrolled=HLPDUnrolledParams(**config.get("unrolled", {})),
            block=HLPDBlockParams(**config.get("block", {})),
            conv_stack=ConvStackParams(**config.get("conv_stack", {})),
            deform=DeformParams(**config.get("deform", {})),
            normalization=NormalizationParams(**config.get("normalization", {})),
            motion=MotionParams.from_dict(config.get("motion") or {}),
            loss=LossParams.from_dict(config.get("loss", {})),
            zero_init=config.get("zero_init", True),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def simulation_ray_trafo_params(data_root: Path) -> RayTrafoParams:
    """The ray transform parameters the data under data_root was simulated with (pinned by data/pipeline.py)."""
    with open(data_root / "config.json") as f:
        ray_cfg = json.load(f)["config"]["ray_trafo"]
    return RayTrafoParams(**{**ray_cfg, "rotAxis": tuple(ray_cfg["rotAxis"])})


def simulation_mu_water(data_root: Path) -> float:
    """The attenuation of water in mm^-1 that the data under data_root was converted from HU with (see data/transforms.py)."""
    with open(data_root / "config.json") as f:
        return float(json.load(f)["config"]["preprocess"]["mu_wa_mm"])


def build_hlpd(config: HLPDConfig, ray_trafo_params: RayTrafoParams, T: int, shape: tuple[int, int, int], device: torch.device, mu_water: float | None = None) -> tuple[HLPDModel, HLPDLoss]:
    """The HLPD model and its loss. Call set_geometry for each study before the forward pass. mu_water (see simulation_mu_water)
    is needed for the 'water' intensity mode."""
    config.block.time_steps = T
    model = HLPDModel(config.unrolled, config.block, config.conv_stack, ray_trafo_params, config.deform, T=T, shape=shape,
                      zero_init=config.zero_init, normalization=config.normalization, mu_water=mu_water, device=device)
    apply_motion_config(model, config.motion, config.conv_stack)

    # NOTE: the extent of the Helmholtz operator is set per study in set_geometry.
    alpha, gamma, beta = config.loss.regularization.helmholtz
    helmholtz_op = HelmholtzOperator(dim=3, alpha=alpha, gamma=gamma, beta=beta, extent=None, device=device)
    # NOTE: the loss shares the model's units, so that it is computed with the same normalization as the iterations.
    loss = HLPDLoss(helmholtz_op, model.units, model.deform, config.loss)
    return model, loss


def set_geometry(model: HLPDModel, loss: HLPDLoss, meta: dict) -> None:
    """Sets the geometry of a study (its physical extent and time steps) in the model's operators and the loss. The ray transforms of
    recently used geometries are cached (see RayTransform._init_ray_transform), so this is cheap when consecutive studies share a geometry."""
    model.set_geometry(meta)
    loss.helmholtz_op.extent = tuple(float(e) for e in model.extent)
