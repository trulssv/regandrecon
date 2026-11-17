"""
The loss of the HLPD model (model/hlpd.py), a joint reconstruction and registration loss of its outputs (w, f, v):

    L =  lambda_data        sum_t D(P_t f_t, y_t)                                                       data
      +  lambda_coupling    sum_t 1/(2K_c) sum_{0<|j|<=K_c} D(f_{t+j} o phi_{t+j -> t}^{-1}, f_t)       coupling
      +  lambda_registered  sum_t 1/J_r sum_{j=1..J_r} D(P_t W(f_{t-j}, phi_{t-j -> t}), y_t)          registered data (optional)
      +  lambda_window      sum_t 1/(2K+1) sum_{i=-K..K} D(P_{t+i} fbar_t, y_{t+i})                    window (optional)
      +  lambda_reg         sum_t ||L v_t||^2                                                           regularization
      +  lambda_temporal    sum_t ||M Delta^n v_t||^2                                                   temporal regularization (optional)
      +  lambda_supervised  sum_t D(W(x_{t-1}, v_{t-1}), x_t)                                          supervised motion (optional, image domain)

Each term has its own parameters (DataTerm, CouplingTerm, RegisteredTerm, WindowTerm, RegularizationTerm, TemporalTerm,
SupervisedMotionTerm below): whether it is included,
its weight lambda, and for all but the regularization its fidelity D. An excluded term is not computed and reported as zero. The
fidelities are means over the voxels (or detector pixels), and the time bins are taken cyclically, where
  - D is the mean squared error ||a - b||^2 ("mse", the default), the mean absolute error ("l1"), or the Huber loss ("huber") with
    threshold huber_delta, scaled by 2 so that it equals the squared error for residuals below huber_delta. L1 and Huber penalize large
    residuals (e.g. edges) less than the squared error, which favours sharper reconstructions over blurred ones. ||.||^2 below is the
    squared error, the default D of every term.
  - f_t is the reconstruction of time bin t and y_t its target: the measured data g_t with P_t = T_t the ray transform of time bin t
    (projection domain, unsupervised), or the ground truth volume x_t with P_t = id (image domain, supervised),
  - w_t = W(f_{t-1}, v_{t-1}) is the prediction of time bin t by the motion from the previous time bin, cyclically (w_0 = W(f_{T-1}, v_{T-1})),
  - phi_{t-j -> t} is the motion from time bin t-j to t, the composition of the motions v_{t-j}, ..., v_{t-1}, so that the j = 1 terms
    are ||w_t - f_t||^2 and ||P_t w_t - y_t||^2, and phi_{t+j -> t} = phi_{t -> t+j}^{-1} for j > 0 the motion back from t+j to t,
  - K_c and J_r are the gaps of the coupling and registered data terms, by default J, the number of time bins needed for complete data
    minus one: the coupling term spans the 2K_c+1 time bins t-K_c, ..., t+K_c,
  - fbar_t is the mean of f_{t-K}, ..., f_{t+K},
  - x_t is the ground truth volume of time bin t (the target in the image domain),
  - L is the Helmholtz operator,
  - Delta^n v_t is the n-th cyclic difference of the velocity fields over the time bins, v_{t+1} - v_t (n = 1) or v_{t+1} - 2 v_t + v_{t-1}
    (n = 2), and M the identity (norm "l2") or L (norm "helmholtz"). The squared norms of both regularizations are means over the voxels.

The loss is the relaxation of the constrained problem
    min sum_t ||P_t f_t - y_t||^2 + R(v)   s.t.  f_t = W(f_{t-1}, v_{t-1}),
with the coupling term as the penalty of the constraint. The constraint for one time bin implies those over several,
f_t = W(f_{t-j}, phi_{t-j -> t}), which the coupling term also penalizes for j <= K_c: in the relaxation, these penalize the drift of the
predictions over several time bins.

The coupling term compares each f_t with its neighbours on both sides, moved to time bin t: the earlier ones forward by the motion,
W(f_{t-j}, phi_{t-j -> t}) = f_{t-j} o phi_{t-j -> t}^{-1}, and the later ones back, f_{t+j} o phi_{t -> t+j}. Each pair of time bins is then
compared in the frames of both, which is symmetric in time and couples each f_t directly, rather than only through the adjoint of a warp.
For a velocity field v constant within a time bin, exp(v)^{-1} = exp(-v), so the motion back is the velocity integrator applied to -v,
the inverse up to interpolation (test/test_inverse_consistency.py).

Each time bin's data covers only part of the arc needed for complete data (90 of 180 degrees for the parallel beam data), so the data
term is blind to the missing wedge of each f_t: the reconstructions keep the artifacts of the limited angle. The data term is the only
data fidelity: the missing wedge of f_t is filled by the coupling term, which pulls it towards the predictions from the K_c time bins on
either side, whose data covers it. By default, J + 1 is the number of consecutive time bins that give complete data
(RayTransform.bins_for_complete_data), e.g. J = 1 for 90 degrees per time bin and J = 3 for 45 degrees, so that each time bin is
coupled directly to all the time bins that complete its data, rather than only through a chain of couplings.

The registered data term instead checks each f_s, moved by the motion, against the data of the next J_r time bins. Under the constraint,
it equals the data term, so in the relaxation it counts the data twice; it is kept for comparison (off by default). Unlike the coupling
term, it drives the motion by the measured data rather than by the reconstructions, whose limited angle artifacts the coupling term also
asks the motion to align. The window term checks the mean of 2K+1 time bins against their data: it is independent of the motion and
biased towards no motion (the true reconstructions do not minimize it), and is meant as a warm-up at most.

The temporal regularization penalizes the change of the velocity fields between consecutive time bins. Each v_t is the stationary velocity
of the motion from time bin t to t+1, so this is a prior on how the motion evolves over the cycle, not on the flow time of a single
motion as in LDDMM. The first difference penalizes the acceleration of the motion and the second its change, which leaves a uniformly
changing velocity free. The fields are compared voxelwise, although v_t lives in the frame of time bin t and v_{t+1} in that of t+1, which
assumes small motions between the time bins. Both penalize sharp transitions of the true motion (e.g. at end of expiration), so the term
is off by default and its weight should be kept small.

The supervised motion term moves the ground truth volumes rather than the reconstructions, so it trains the motion independently of the
quality of the reconstructions (and of their limited angle artifacts). It needs the ground truth volumes, i.e. the image domain. It is
off by default, and meant to check whether the motion nets can learn the motion at all, e.g. in a motion stage of the training (see
train/stages.py), rather than as part of the loss of the joint problem.

The data term ties each f_t to its own data. Only then does the coupling term drive the motion: without it, f_t could follow w_t for any
v, including v = 0. The factor 1/2 of the coupling energy is absorbed in lambda_coupling.

All terms but the regularizations are computed in the units of the HLPD iterations (model.hlpd.IterationUnits, shared with the model),
i.e. the images are c x and the data s_t c g_t with the intensity scale c and the operator scales s_t = 1 / ||T_t|| of the model's
NormalizationParams. With both normalizations, the data and coupling terms are then of the same order (O(1)), rather than four orders
of magnitude apart in physical units. Without normalization, c = s_t = 1 and the loss is in physical units. The velocities are in mm in
both cases.
"""
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch.nn.functional import huber_loss, l1_loss, mse_loss

from model.hlpd import IterationUnits
from operators.lddmm.deform import FlowDeformationOperator
from operators.lddmm.helmholtz import HelmholtzOperator
from tracking.timeline import region


FIDELITIES = ("mse", "l1", "huber")


def fidelity(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse", huber_delta: float = 0.1) -> torch.Tensor:
    """The fidelity D(pred, target) of the module docstring, a mean over the elements."""
    if kind == "l1":
        return l1_loss(pred, target)
    if kind == "huber":
        return 2 * huber_loss(pred, target, delta=huber_delta)
    return mse_loss(pred, target)


@dataclass
class _Term:
    include: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        assert self.weight >= 0, f"The weight of a loss term must be non-negative, but got {self.weight}."


@dataclass
class _FidelityTerm(_Term):
    fidelity: str = "mse"           # "mse", "l1" or "huber", see fidelity
    huber_delta: float = 0.1        # threshold of the Huber loss, in the units of the iterations (water units with intensity "water": 0.1 = 100 HU)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.fidelity in FIDELITIES, f"Invalid fidelity {self.fidelity}. Expected one of {FIDELITIES}."
        assert self.huber_delta > 0, f"huber_delta must be positive, but got {self.huber_delta}."

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return fidelity(pred, target, self.fidelity, self.huber_delta)


@dataclass
class DataTerm(_FidelityTerm):
    """Each reconstruction against its own target. The only data fidelity of each f_t, so it should only be excluded for experiments."""


@dataclass
class CouplingTerm(_FidelityTerm):
    """Each reconstruction against its neighbours moved by the motion, from the previous and back from the next time bins."""
    gaps: int | None = None         # K_c, the number of time bins on either side (2K_c+1 in all), None for the number the geometry needs for complete data

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.gaps is None or self.gaps >= 1, f"The coupling gaps must be at least 1, but got {self.gaps}."


@dataclass
class RegisteredTerm(_FidelityTerm):
    """The reconstructions moved to the next time bins against their data. Counts the data twice, so it is excluded by default."""
    include: bool = False
    gaps: int | None = None         # J_r, the number of next time bins, None for the number the geometry needs for complete data

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.gaps is None or self.gaps >= 1, f"The registered gaps must be at least 1, but got {self.gaps}."


@dataclass
class WindowTerm(_FidelityTerm):
    """The mean of 2k+1 time bins against their data. Biased towards no motion, a warm-up at most, so it is excluded by default."""
    include: bool = False
    k: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.k >= 1, f"The window k must be at least 1, but got {self.k}."


@dataclass
class RegularizationTerm(_Term):
    """The Helmholtz regularization ||L v_t||^2 of the velocity fields."""
    weight: float = 1e-6
    helmholtz: tuple[float, float, float] = (1.0, 1.0, 1.0)     # (alpha, gamma, beta) of the Helmholtz operator L

    def __post_init__(self) -> None:
        super().__post_init__()
        self.helmholtz = tuple(self.helmholtz)  # type: ignore[assignment]


@dataclass
class TemporalTerm(_Term):
    """The temporal regularization ||M Delta^n v_t||^2 of the velocity fields over the time bins. Off by default."""
    include: bool = False
    weight: float = 1e-3
    order: int = 1                  # n, 1 for the first difference v_{t+1} - v_t, 2 for the second v_{t+1} - 2 v_t + v_{t-1}
    norm: str = "l2"                # M, "l2" for the identity or "helmholtz" for the Helmholtz operator L of the regularization term

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.order in (1, 2), f"The order of the temporal regularization must be 1 or 2, but got {self.order}."
        assert self.norm in ("l2", "helmholtz"), f"Invalid temporal norm {self.norm}. Expected 'l2' or 'helmholtz'."


@dataclass
class SupervisedMotionTerm(_FidelityTerm):
    """The ground truth volumes moved by the motion against those of the next time bins. Image domain only, off by default."""
    include: bool = False


@dataclass
class LossParams:
    """The loss config: the domain and one section per term, e.g. (omitted sections and keys take the defaults above)
        {"domain": "image",
         "data": {"include": true, "weight": 1.0, "fidelity": "huber", "huber_delta": 0.1},
         "coupling": {"include": true, "weight": 5.0, "fidelity": "mse", "gaps": 3},
         "registered": {"include": false}, "window": {"include": false},
         "regularization": {"include": true, "weight": 1e-3, "helmholtz": [1.0, 1.0, 1.0]},
         "temporal": {"include": true, "weight": 1e-3, "order": 1, "norm": "l2"},
         "supervised_motion": {"include": false}}"""
    domain: str = "image"           # "image" (supervised, needs volumes) or "projection" (unsupervised)
    data: DataTerm = None           # type: ignore[assignment]
    coupling: CouplingTerm = None   # type: ignore[assignment]
    registered: RegisteredTerm = None  # type: ignore[assignment]
    window: WindowTerm = None       # type: ignore[assignment]
    regularization: RegularizationTerm = None  # type: ignore[assignment]
    temporal: TemporalTerm = None   # type: ignore[assignment]
    supervised_motion: SupervisedMotionTerm = None  # type: ignore[assignment]

    TERMS: ClassVar[dict[str, type]] = {"data": DataTerm, "coupling": CouplingTerm, "registered": RegisteredTerm, "window": WindowTerm,
                                        "regularization": RegularizationTerm, "temporal": TemporalTerm, "supervised_motion": SupervisedMotionTerm}

    def __post_init__(self) -> None:
        assert self.domain in ("image", "projection"), f"Invalid loss domain {self.domain}. Expected 'image' or 'projection'."
        for name, cls in self.TERMS.items():
            term = getattr(self, name)
            setattr(self, name, cls() if term is None else cls(**term) if isinstance(term, dict) else term)
        assert not self.supervised_motion.include or self.domain == "image", "The supervised motion term needs the image domain (the ground truth volumes)."

    @classmethod
    def from_dict(cls, config: dict) -> "LossParams":
        """From the config above, or from the flat config of earlier runs (lambda_coupling, coupling_gaps, ..., e.g. of their checkpoints)."""
        legacy = ("lambda_coupling", "coupling_gaps", "lambda_registered", "registered_gaps", "lambda_window", "window_k", "lambda_reg", "data_loss",
                  "huber_delta", "helmholtz")
        if not any(key in config for key in legacy):
            return cls(**config)
        fid = {"fidelity": config.get("data_loss", "mse"), "huber_delta": config.get("huber_delta", 0.1)}
        lambda_registered, lambda_window = config.get("lambda_registered", 0.0), config.get("lambda_window", 0.0)
        return cls(
            domain=config.get("domain", "image"),
            data=DataTerm(**fid),
            # NOTE: the coupling term was always computed, also with a zero weight.
            coupling=CouplingTerm(weight=config.get("lambda_coupling", 1.0), gaps=config.get("coupling_gaps")),
            registered=RegisteredTerm(include=lambda_registered != 0, weight=lambda_registered or 1.0, gaps=config.get("registered_gaps"), **fid),
            window=WindowTerm(include=lambda_window != 0, weight=lambda_window or 1.0, k=config.get("window_k", 1), **fid),
            regularization=RegularizationTerm(weight=config.get("lambda_reg", 1e-6), helmholtz=tuple(config.get("helmholtz", (1.0, 1.0, 1.0)))),
        )


@dataclass
class HLPDLossTerms:
    data_loss: torch.Tensor
    coupling_loss: torch.Tensor
    registered_loss: torch.Tensor   # zero if the term is excluded (the default)
    window_loss: torch.Tensor       # zero if the term is excluded (the default)
    regularization_loss: torch.Tensor
    temporal_loss: torch.Tensor     # zero if the term is excluded (the default)
    supervised_motion_loss: torch.Tensor  # zero if the term is excluded (the default)
    total_loss: torch.Tensor


class HLPDLoss(torch.nn.Module):
    """The joint reconstruction and registration loss above, in the units of the HLPD iterations. units is the model's IterationUnits
    (HLPDModel.units) and deform its deformation operator (HLPDModel.deform), and params the domain ("projection": data terms against the
    measured data, or "image": against the ground truth volumes) and the terms, see LossParams. The coupling and registered gaps are K_c
    and J_r, None for J, the number of time bins the geometry needs for complete data minus one."""

    def __init__(self, helmholtz_op: HelmholtzOperator, units: IterationUnits, deform: FlowDeformationOperator, params: LossParams | None = None) -> None:
        super().__init__()
        params = params if params is not None else LossParams()
        self.helmholtz_op = helmholtz_op
        self.units = units
        self.deform = deform
        self.domain = params.domain
        self._set_terms(params)

    def _set_terms(self, params: LossParams) -> None:
        self.data, self.coupling, self.registered = params.data, params.coupling, params.registered
        self.window, self.regularization, self.temporal = params.window, params.regularization, params.temporal
        self.supervised_motion = params.supervised_motion

    def set_params(self, params: LossParams) -> None:
        """Changes the terms, e.g. for a stage of the training (see train/stages.py). The domain and the Helmholtz parameters cannot
        change, as the targets and the Helmholtz operator depend on them."""
        assert params.domain == self.domain, f"The loss domain cannot change from {self.domain} to {params.domain}."
        assert tuple(params.regularization.helmholtz) == tuple(self.regularization.helmholtz), \
            f"The Helmholtz parameters cannot change from {self.regularization.helmholtz} to {params.regularization.helmholtz}."
        self._set_terms(params)

    def _complete_data_gaps(self) -> int:
        """J, the number of time bins needed for complete data minus one (at least 1)."""
        return max(self.units.ray_trafo.ray_trafo.bins_for_complete_data() - 1, 1)

    def n_coupling_gaps(self) -> int:
        """K_c, the number of time bins on either side whose predictions are coupled to each time bin."""
        return self.coupling.gaps if self.coupling.gaps is not None else self._complete_data_gaps()

    def n_registered_gaps(self) -> int:
        """J_r, the number of preceding time bins whose reconstructions are checked against the data of each time bin."""
        return self.registered.gaps if self.registered.gaps is not None else self._complete_data_gaps()

    def _predict(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """P_t x in the units of the iterations, for an image x in mm^-1."""
        return self.units.project_bin(x, t) if self.domain == "projection" else self.units.images([x])[0]

    def _grid(self, v: torch.Tensor) -> torch.Tensor:
        """The sampling grid of W(., v), i.e. W(f, v) = f sampled at the grid, as computed by the deformation operator."""
        return self.deform.velocity_integrator(self.deform._normalize(v))

    def _inverse_grid(self, v: torch.Tensor) -> torch.Tensor:
        """The sampling grid that undoes W(., v), exp(v) rather than exp(-v): the integrator applied to -v (_normalize commutes with it)."""
        return self.deform.velocity_integrator(self.deform._normalize(-v))

    def _moved(self, w: list[torch.Tensor], f: list[torch.Tensor], v: list[torch.Tensor], J: int) -> list[list[torch.Tensor]]:
        """moved[t][j - 1] = W(f_{t-j}, phi_{t-j -> t}) in mm^-1 for j = 1..J, with moved[t][0] = w_t the prediction of the model."""
        T = len(f)
        assert J < T, f"The coupling and registered data terms need fewer than T = {T} gaps, but got {J}."
        moved = [[w[t]] for t in range(T)]
        if J > 1:
            # NOTE: the motions are composed and each f_{t-j} is warped once, rather than warping w_{t-1} on, which would interpolate it j times.
            # W(W(f, v_a), v_b) = f sampled at grid_a o grid_b, so the grid from t-j to t is grid_{t-j} o ... o grid_{t-1}.
            with region("loss/coupling/deform"):
                grids = [self._grid(vt) for vt in v]
                for t in range(T):
                    grid = grids[(t - 1) % T]
                    for j in range(2, J + 1):
                        grid = self.deform.velocity_integrator._compose(grids[(t - j) % T], grid)
                        moved[t].append(self.deform.group_action(f[(t - j) % T], grid).squeeze(1))
        return moved

    def _moved_back(self, f: list[torch.Tensor], v: list[torch.Tensor], K: int) -> list[list[torch.Tensor]]:
        """moved_back[t][j - 1] = f_{t+j} o phi_{t -> t+j} in mm^-1 for j = 1..K, f_{t+j} moved back to time bin t."""
        T = len(f)
        with region("loss/coupling/deform"):
            # NOTE: f_{t+j} o phi_{t -> t+j} is f_{t+j} sampled at phi_{t+j-1} o ... o phi_t, so each motion further on is composed on the left.
            grids = [self._inverse_grid(vt) for vt in v]
            moved_back = [[] for _ in range(T)]
            for t in range(T):
                grid = grids[t]
                for j in range(1, K + 1):
                    if j > 1:
                        grid = self.deform.velocity_integrator._compose(grids[(t + j - 1) % T], grid)
                    moved_back[t].append(self.deform.group_action(f[(t + j) % T], grid).squeeze(1))
        return moved_back

    def _coupling_loss(self, moved: list[list[torch.Tensor]], moved_back: list[list[torch.Tensor]], f: list[torch.Tensor]) -> torch.Tensor:
        K = self.n_coupling_gaps()
        f_units = self.units.images(f)
        terms = [self.coupling(m, f_units[t]) for t in range(len(f)) for m in self.units.images(moved[t][:K] + moved_back[t][:K])]
        return torch.stack(terms).sum() / (2 * K)

    def _registered_loss(self, moved: list[list[torch.Tensor]], target: list[torch.Tensor]) -> torch.Tensor:
        J = self.n_registered_gaps()
        terms = [self.registered(self._predict(m, t), target[t]) for t in range(len(moved)) for m in moved[t][:J]]
        return torch.stack(terms).sum() / J

    def _window_loss(self, f: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
        T, K = len(f), self.window.k
        assert 2 * K + 1 <= T, f"The window of 2K+1 = {2 * K + 1} time bins is longer than the T = {T} time bins."
        terms = []
        for t in range(T):
            fbar = torch.stack([f[(t + i) % T] for i in range(-K, K + 1)]).mean(0)
            terms += [self.window(self._predict(fbar, (t + i) % T), target[(t + i) % T]) for i in range(-K, K + 1)]
        return torch.stack(terms).sum() / (2 * K + 1)

    def _temporal_loss(self, v: list[torch.Tensor]) -> torch.Tensor:
        T = len(v)
        assert T >= self.temporal.order + 1, f"The temporal regularization of order {self.temporal.order} needs more than T = {T} time bins."
        if self.temporal.order == 1:
            diffs = [v[(t + 1) % T] - v[t] for t in range(T)]
        else:
            diffs = [v[(t + 1) % T] - 2 * v[t] + v[(t - 1) % T] for t in range(T)]
        # NOTE: L is linear, so the Helmholtz norm of the difference is that of the difference of the L v_t.
        norm = self.helmholtz_op if self.temporal.norm == "helmholtz" else (lambda d: d)
        return torch.stack([norm(d).square().mean() for d in diffs]).sum()

    def _supervised_motion_loss(self, v: list[torch.Tensor], x: list[torch.Tensor]) -> torch.Tensor:
        T = len(v)
        moved = [self.deform.group_action(x[(t - 1) % T], self._grid(v[(t - 1) % T])).squeeze(1) for t in range(T)]
        x_units = self.units.images(x)
        return torch.stack([self.supervised_motion(m, x_units[t]) for t, m in enumerate(self.units.images(moved))]).sum()

    def forward(self, w: list[torch.Tensor], f: list[torch.Tensor], v: list[torch.Tensor], y: list[torch.Tensor]) -> HLPDLossTerms:
        """The physical outputs of the model, w, f: T tensors (B, D, H, W) in mm^-1 and v: T tensors (B, D, H, W, 3) in mm, and the
        physical targets y: T volumes (B, D, H, W) in mm^-1 (image domain) or T measured data (B, nAngles, nCol, nRow) (projection domain)."""
        assert len(w) == len(f) == len(v) == len(y), f"Expected the same number of time bins, but got {len(w)}, {len(f)}, {len(v)} and {len(y)}."
        zero = f[0].new_zeros(())
        data_loss = coupling_loss = registered_loss = window_loss = regularization_loss = temporal_loss = supervised_motion_loss = zero

        # NOTE: one region per term, which do not overlap, so that the timing plot can sum them (see tracking/timeline.py). An excluded
        # term gets no region.
        target = self.units.data(y) if self.domain == "projection" else self.units.images(y)
        if self.data.include:
            with region("loss/data"):
                data_loss = torch.stack([self.data(self._predict(ft, t), target[t]) for t, ft in enumerate(f)]).sum()
        if self.coupling.include or self.registered.include:
            with region("loss/coupling"):
                # NOTE: the registered data term reuses the moved reconstructions, so their warps are timed under the coupling term.
                K = self.n_coupling_gaps() if self.coupling.include else 1
                assert 2 * K + 1 <= len(f), f"The coupling term spans 2K_c+1 = {2 * K + 1} time bins, more than the T = {len(f)} time bins."
                moved = self._moved(w, f, v, max(K, self.n_registered_gaps() if self.registered.include else 1))
                if self.coupling.include:
                    coupling_loss = self._coupling_loss(moved, self._moved_back(f, v, K), f)
        if self.registered.include:
            with region("loss/registered"):
                registered_loss = self._registered_loss(moved, target)
        if self.window.include:
            with region("loss/window"):
                window_loss = self._window_loss(f, target)
        if self.regularization.include:
            with region("loss/helmholtz"):
                regularization_loss = torch.stack([self.helmholtz_op(vt).square().mean() for vt in v]).sum()
        if self.temporal.include:
            with region("loss/temporal"):
                temporal_loss = self._temporal_loss(v)
        if self.supervised_motion.include:
            with region("loss/supervised_motion"):
                supervised_motion_loss = self._supervised_motion_loss(v, y)

        total_loss = (self.data.weight * data_loss + self.coupling.weight * coupling_loss + self.registered.weight * registered_loss
                      + self.window.weight * window_loss + self.regularization.weight * regularization_loss + self.temporal.weight * temporal_loss
                      + self.supervised_motion.weight * supervised_motion_loss)
        return HLPDLossTerms(data_loss=data_loss, coupling_loss=coupling_loss, registered_loss=registered_loss, window_loss=window_loss,
                             regularization_loss=regularization_loss, temporal_loss=temporal_loss,
                             supervised_motion_loss=supervised_motion_loss, total_loss=total_loss)
