"""
The loss of the HLPD model (model/hlpd.py), a joint reconstruction and registration loss of its outputs (w, f, v):

    L =  sum_t ||P_t f_t - y_t||^2                                                       data
      +  lambda_coupling    sum_t ||w_t - f_t||^2                                        coupling
      +  lambda_registered  sum_t 1/J sum_{j=1..J} ||P_t W(f_{t-j}, phi_{t-j -> t}) - y_t||^2  registered data
      +  lambda_window      sum_t 1/(2K+1) sum_{i=-K..K} ||P_{t+i} fbar_t - y_{t+i}||^2  window (optional)
      +  lambda_reg         sum_t ||L v_t||^2                                            regularization

with the norms as means over the voxels (or detector pixels) and the time bins taken cyclically, where
  - f_t is the reconstruction of time bin t and y_t its target: the measured data g_t with P_t = T_t the ray transform of time bin t
    (projection domain, unsupervised), or the ground truth volume x_t with P_t = id (image domain, supervised),
  - w_t = W(f_{t-1}, v_{t-1}) is the prediction of time bin t by the motion from the previous time bin, cyclically (w_0 = W(f_{T-1}, v_{T-1})),
  - phi_{t-j -> t} is the motion from time bin t-j to t, the composition of the motions v_{t-j}, ..., v_{t-1}, so that the j = 1 term is
    ||P_t w_t - y_t||^2,
  - fbar_t is the mean of f_{t-K}, ..., f_{t+K},
  - L is the Helmholtz operator.

Each time bin's data covers only part of the arc needed for complete data (90 of 180 degrees for the parallel beam data), so the data
term is blind to the missing wedge of each f_t: the reconstructions keep the artifacts of the limited angle. The registered data term
checks each f_s, moved by the motion, against the data of the next J time bins, which together with its own cover the complete arc.
By default, J + 1 is the number of consecutive time bins that give complete data (RayTransform.bins_for_complete_data), e.g. J = 1
for 90 degrees per time bin and J = 3 for 45 degrees. It is minimized by the true reconstructions with the true motion, and so also
rewards the motion. The window term instead checks the mean of 2K+1 time bins against their data: it is independent of the motion
and biased towards no motion (the true reconstructions do not minimize it), and is meant as a warm-up at most.

The data term ties each f_t to its own data. Only then does the coupling term, which compares f_t to its prediction w_t, drive the
motion: without it, f_t could follow w_t for any v, including v = 0. The factor 1/2 of the coupling energy is absorbed in lambda_coupling.

All terms but the regularization are computed in the units of the HLPD iterations (model.hlpd.IterationUnits, shared with the model),
i.e. the images are c x and the data s_t c g_t with the intensity scale c and the operator scales s_t = 1 / ||T_t|| of the model's
NormalizationParams. With both normalizations, the data and coupling terms are then of the same order (O(1)), rather than four orders
of magnitude apart in physical units. Without normalization, c = s_t = 1 and the loss is in physical units. The velocities are in mm in
both cases.
"""
from dataclasses import dataclass
import torch
from torch.nn.functional import mse_loss

from model.hlpd import IterationUnits
from operators.lddmm.deform import FlowDeformationOperator
from operators.lddmm.helmholtz import HelmholtzOperator
from tracking.timeline import region


@dataclass
class HLPDLossTerms:
    data_loss: torch.Tensor
    coupling_loss: torch.Tensor
    registered_loss: torch.Tensor   # zero if lambda_registered is 0
    window_loss: torch.Tensor       # zero if lambda_window is 0
    regularization_loss: torch.Tensor
    total_loss: torch.Tensor


class HLPDLoss(torch.nn.Module):
    """The joint reconstruction and registration loss above, in the units of the HLPD iterations. units is the model's IterationUnits
    (HLPDModel.units) and deform its deformation operator (HLPDModel.deform), and domain "projection" (data terms against the measured
    data) or "image" (against the ground truth volumes). registered_gaps is J, None for the number of time bins the geometry needs."""

    def __init__(self, helmholtz_op: HelmholtzOperator, units: IterationUnits, deform: FlowDeformationOperator, domain: str = "projection",
                 lambda_coupling: float = 1.0, lambda_registered: float = 1.0, registered_gaps: int | None = None,
                 lambda_window: float = 0.0, window_k: int = 1, lambda_reg: float = 0.1) -> None:
        super().__init__()
        assert domain in ("image", "projection"), f"Invalid loss domain {domain}. Expected 'image' or 'projection'."
        assert registered_gaps is None or registered_gaps >= 1, f"registered_gaps must be at least 1, but got {registered_gaps}."
        assert window_k >= 1, f"window_k must be at least 1, but got {window_k}."
        self.helmholtz_op = helmholtz_op
        self.units = units
        self.deform = deform
        self.domain = domain
        self.lambda_coupling = lambda_coupling
        self.lambda_registered = lambda_registered
        self.registered_gaps = registered_gaps
        self.lambda_window = lambda_window
        self.window_k = window_k
        self.lambda_reg = lambda_reg

    def gaps(self) -> int:
        """J, the number of preceding time bins whose reconstructions are checked against the data of each time bin."""
        if self.registered_gaps is not None:
            return self.registered_gaps
        return max(self.units.ray_trafo.ray_trafo.bins_for_complete_data() - 1, 1)

    def _predict(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """P_t x in the units of the iterations, for an image x in mm^-1."""
        return self.units.project_bin(x, t) if self.domain == "projection" else self.units.images([x])[0]

    def _grid(self, v: torch.Tensor) -> torch.Tensor:
        """The sampling grid of W(., v), i.e. W(f, v) = f sampled at the grid, as computed by the deformation operator."""
        return self.deform.velocity_integrator(self.deform._normalize(v))

    def _registered_loss(self, w: list[torch.Tensor], f: list[torch.Tensor], v: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
        T, J = len(f), self.gaps()
        assert J < T, f"The registered data term needs fewer than T = {T} gaps, but got {J}."
        # j = 1: the prediction w_t = W(f_{t-1}, v_{t-1}) of the model
        terms = [mse_loss(self._predict(w[t], t), target[t]) for t in range(T)]
        if J > 1:
            # NOTE: the motions are composed and each f_{t-j} is warped once, rather than warping w_{t-1} on, which would interpolate it j times.
            # W(W(f, v_a), v_b) = f sampled at grid_a o grid_b, so the grid from t-j to t is grid_{t-j} o ... o grid_{t-1}.
            with region("loss/registered/deform"):
                grids = [self._grid(vt) for vt in v]
                for t in range(T):
                    grid = grids[(t - 1) % T]
                    for j in range(2, J + 1):
                        grid = self.deform.velocity_integrator._compose(grids[(t - j) % T], grid)
                        warped = self.deform.group_action(f[(t - j) % T], grid).squeeze(1)
                        terms.append(mse_loss(self._predict(warped, t), target[t]))
        return torch.stack(terms).sum() / J

    def _window_loss(self, f: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
        T, K = len(f), self.window_k
        assert 2 * K + 1 <= T, f"The window of 2K+1 = {2 * K + 1} time bins is longer than the T = {T} time bins."
        terms = []
        for t in range(T):
            fbar = torch.stack([f[(t + i) % T] for i in range(-K, K + 1)]).mean(0)
            terms += [mse_loss(self._predict(fbar, (t + i) % T), target[(t + i) % T]) for i in range(-K, K + 1)]
        return torch.stack(terms).sum() / (2 * K + 1)

    def forward(self, w: list[torch.Tensor], f: list[torch.Tensor], v: list[torch.Tensor], y: list[torch.Tensor]) -> HLPDLossTerms:
        """The physical outputs of the model, w, f: T tensors (B, D, H, W) in mm^-1 and v: T tensors (B, D, H, W, 3) in mm, and the
        physical targets y: T volumes (B, D, H, W) in mm^-1 (image domain) or T measured data (B, nAngles, nCol, nRow) (projection domain)."""
        assert len(w) == len(f) == len(v) == len(y), f"Expected the same number of time bins, but got {len(w)}, {len(f)}, {len(v)} and {len(y)}."
        zero = f[0].new_zeros(())

        # NOTE: one region per term, which do not overlap, so that the timing plot can sum them (see tracking/timeline.py). A term that
        # is switched off gets no region.
        target = self.units.data(y) if self.domain == "projection" else self.units.images(y)
        with region("loss/data"):
            data_loss = torch.stack([mse_loss(self._predict(ft, t), target[t]) for t, ft in enumerate(f)]).sum()
        with region("loss/coupling"):
            coupling_loss = torch.stack([mse_loss(wt, ft) for wt, ft in zip(self.units.images(w), self.units.images(f))]).sum()
        registered_loss = window_loss = zero
        if self.lambda_registered != 0:
            with region("loss/registered"):
                registered_loss = self._registered_loss(w, f, v, target)
        if self.lambda_window != 0:
            with region("loss/window"):
                window_loss = self._window_loss(f, target)
        with region("loss/helmholtz"):
            regularization_loss = torch.stack([self.helmholtz_op(vt).square().mean() for vt in v]).sum()

        total_loss = (data_loss + self.lambda_coupling * coupling_loss + self.lambda_registered * registered_loss
                      + self.lambda_window * window_loss + self.lambda_reg * regularization_loss)
        return HLPDLossTerms(data_loss=data_loss, coupling_loss=coupling_loss, registered_loss=registered_loss, window_loss=window_loss,
                             regularization_loss=regularization_loss, total_loss=total_loss)
