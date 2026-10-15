"""
The loss of the HLPD model (model/hlpd.py), a joint reconstruction and registration loss of its outputs (w, f, v):

    L = sum_t ||P f_t - y_t||^2  +  lambda_coupling sum_t ||w_t - f_t||^2  +  lambda_reg sum_t ||L v_t||^2

with the norms as means over the voxels (or detector pixels), where
  - f_t is the reconstruction of time bin t and y_t its target: the measured data g_t with P = T_t the ray transform of time bin t
    (projection domain, unsupervised), or the ground truth volume x_t with P = id (image domain, supervised),
  - w_t = W(f_{t-1}, v_{t-1}) is the prediction of time bin t by the motion from the previous time bin, cyclically (w_0 = W(f_{T-1}, v_{T-1})),
  - L is the Helmholtz operator.

All terms but the regularization are computed in the units of the HLPD iterations (model.hlpd.IterationUnits, shared with the model),
i.e. the images are c x and the data s_t c g_t with the intensity scale c and the operator scales s_t = 1 / ||T_t|| of the model's
NormalizationParams. With both normalizations, the data and coupling terms are then of the same order (O(1)), rather than four orders
of magnitude apart in physical units. Without normalization, c = s_t = 1 and the loss is in physical units. The velocities are in mm in
both cases.

The data term ties each f_t to its own data. Only then does the coupling term, which compares f_t to its prediction w_t, drive the
motion: without it, f_t could follow w_t for any v, including v = 0. The factor 1/2 of the coupling energy is absorbed in lambda_coupling.
"""
from dataclasses import dataclass
import torch
from torch.nn.functional import mse_loss

from model.hlpd import IterationUnits
from operators.lddmm.helmholtz import HelmholtzOperator
from tracking.timeline import region


@dataclass
class HLPDLossTerms:
    data_loss: torch.Tensor
    coupling_loss: torch.Tensor
    regularization_loss: torch.Tensor
    total_loss: torch.Tensor


class HLPDLoss(torch.nn.Module):
    """The joint reconstruction and registration loss above, in the units of the HLPD iterations. units is the model's IterationUnits
    (HLPDModel.units), and domain "projection" (data term against the measured data) or "image" (against the ground truth volumes)."""

    def __init__(self, helmholtz_op: HelmholtzOperator, units: IterationUnits, domain: str = "projection", lambda_coupling: float = 1.0, lambda_reg: float = 0.1) -> None:
        super().__init__()
        assert domain in ("image", "projection"), f"Invalid loss domain {domain}. Expected 'image' or 'projection'."
        self.helmholtz_op = helmholtz_op
        self.units = units
        self.domain = domain
        self.lambda_coupling = lambda_coupling
        self.lambda_reg = lambda_reg

    def forward(self, w: list[torch.Tensor], f: list[torch.Tensor], v: list[torch.Tensor], y: list[torch.Tensor]) -> HLPDLossTerms:
        """The physical outputs of the model, w, f: T tensors (B, D, H, W) in mm^-1 and v: T tensors (B, D, H, W, 3) in mm, and the
        physical targets y: T volumes (B, D, H, W) in mm^-1 (image domain) or T measured data (B, nAngles, nCol, nRow) (projection domain)."""
        assert len(w) == len(f) == len(v) == len(y), f"Expected the same number of time bins, but got {len(w)}, {len(f)}, {len(v)} and {len(y)}."

        if self.domain == "projection":
            with region("loss/ray_trafo"):
                prediction = self.units.project(f)
            target = self.units.data(y)
        else:
            prediction, target = self.units.images(f), self.units.images(y)
        data_loss = torch.stack([mse_loss(p, yt) for p, yt in zip(prediction, target)]).sum()
        coupling_loss = torch.stack([mse_loss(wt, ft) for wt, ft in zip(self.units.images(w), self.units.images(f))]).sum()
        with region("loss/helmholtz"):
            regularization_loss = torch.stack([self.helmholtz_op(vt).square().mean() for vt in v]).sum()

        total_loss = data_loss + self.lambda_coupling * coupling_loss + self.lambda_reg * regularization_loss
        return HLPDLossTerms(data_loss=data_loss, coupling_loss=coupling_loss, regularization_loss=regularization_loss, total_loss=total_loss)
