"""
Builds the HLPD model and its loss (model/hlpd_loss.py) from a config, shared by the training scripts (train/train.py, test/train_hlpd_overfit.py).

The config holds the architecture and loss parameters, as dicts of the dataclasses in model/hlpd.py and LossParams below:
    {"unrolled": {...}, "block": {...}, "conv_stack": {...}, "deform": {...}, "normalization": {...}, "loss": {...}, "zero_init": bool}
The ray transform is not part of it: it must match the geometry the data was simulated with, see simulation_ray_trafo_params.
"""
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from model.hlpd import ConvStackParams, DeformParams, HLPDBlockParams, HLPDModel, HLPDUnrolledParams, NormalizationParams, RayTrafoParams
from model.hlpd_loss import HLPDLoss
from operators.lddmm.helmholtz import HelmholtzOperator

logger = logging.getLogger(__name__)


def configure_cuda(exact_fp32: bool = False) -> None:
    """Settings for the training scripts, to be called before anything is allocated on the GPU (see model/PERFORMANCE.md):
    - expandable segments in the caching allocator, which avoids most of the fragmentation of the large, differently sized
      activations of the volume and sinogram nets.
    - cuDNN autotuning, which picks the fastest convolution algorithms for the (fixed) volume and sinogram shapes on the first
      step (about 25% faster steps at full resolution). The chosen algorithms may differ between runs.
    - exact_fp32: full fp32 instead of TF32 convolutions (cuDNN's default). TF32's 10 bit mantissa is amplified by the large
      sinogram values over the recurrent updates: after one forward pass at full resolution, f differed by 0.3% from fp32, and by
      1.4% between two TF32 algorithms, so with autotuning TF32 results also depend on the algorithms that were picked. fp32 is
      bitwise independent of the algorithm, but makes a step about 45% slower."""
    if torch.cuda.is_initialized():
        logger.warning("configure_cuda was called after CUDA was initialized, so the allocator settings have no effect.")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True
    if exact_fp32:
        torch.backends.cudnn.allow_tf32 = False


@dataclass
class LossParams:
    domain: str = "image"                                               # "image" (supervised, needs volumes) or "projection" (unsupervised)
    lambda_coupling: float = 1.0                                        # weight of the coupling ||w_t - f_t||^2 of the reconstructions and their predictions by the motion
    lambda_reg: float = 1e-6                                            # weight of the Helmholtz regularization of the velocity fields
    helmholtz: tuple[float, float, float] = (1.0, 1.0, 1.0)             # (alpha, gamma, beta) of the Helmholtz operator

    def __post_init__(self) -> None:
        assert self.domain in ("image", "projection"), f"Invalid loss domain {self.domain}. Expected 'image' or 'projection'."
        self.helmholtz = tuple(self.helmholtz)  # type: ignore[assignment]


@dataclass
class HLPDConfig:
    unrolled: HLPDUnrolledParams = field(default_factory=HLPDUnrolledParams)
    block: HLPDBlockParams = field(default_factory=HLPDBlockParams)
    conv_stack: ConvStackParams = field(default_factory=ConvStackParams)
    deform: DeformParams = field(default_factory=DeformParams)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
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
            loss=LossParams(**config.get("loss", {})),
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

    # NOTE: the extent of the Helmholtz operator is set per study in set_geometry.
    alpha, gamma, beta = config.loss.helmholtz
    helmholtz_op = HelmholtzOperator(dim=3, alpha=alpha, gamma=gamma, beta=beta, extent=None, device=device)
    # NOTE: the loss shares the model's units, so that it is computed with the same normalization as the iterations.
    loss = HLPDLoss(helmholtz_op, model.units, domain=config.loss.domain, lambda_coupling=config.loss.lambda_coupling, lambda_reg=config.loss.lambda_reg)
    return model, loss


def set_geometry(model: HLPDModel, loss: HLPDLoss, meta: dict) -> None:
    """Sets the geometry of a study (its physical extent and time steps) in the model's operators and the loss. The ray transforms of
    recently used geometries are cached (see RayTransform._init_ray_transform), so this is cheap when consecutive studies share a geometry."""
    model.set_geometry(meta)
    loss.helmholtz_op.extent = tuple(float(e) for e in model.extent)
