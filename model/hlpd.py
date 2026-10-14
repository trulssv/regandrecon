from typing import Tuple, List, overload, Literal, cast
from dataclasses import dataclass, asdict
# torch imports

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tracking.timeline import region

import copy


## --------------- LOCAL IMPORTS ---------------

from operators.tomo.ray_trafo import RayTransform, operator_norm
from operators.lddmm.deform import FlowDeformationOperator
from data.utils import study_extent

def _validate_input(x: Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...]) -> None:
    """Checks that the input is a tuple (f, h, v) of either lists of torch.Tensor or torch.Tensor. """
    if not isinstance(x, tuple) or len(x) != 3:
        raise ValueError("Input must be a tuple of three elements.")
    if all(isinstance(elem, list) for elem in x):
        if not all(isinstance(sub_elem, torch.Tensor) for sublist in x for sub_elem in sublist):
            raise ValueError("All elements in the input lists must be torch.Tensor.")
    elif all(isinstance(elem, torch.Tensor) for elem in x):
        pass
    else:
        raise ValueError("Input must be either a tuple of lists of torch.Tensor or a tuple of torch.Tensor.")



@dataclass
class HLPDUnrolledParams:
    num_iters: int = 7                                                          # The number of iterations to apply the HLPD iteration model.
    shared: bool = False                                                        # Whether the HLPD iteration model is shared across iterations.
    grad_checkpointing: bool = False                                            # Whether to use gradient checkpointing for the HLPD iteration model.

@dataclass
class HLPDBlockParams:
    recurrent: bool = False                                                      # Whether the model is recurrent in time
    time_steps: int = 10
    shared: bool = False                                                        # Recurrent mode only: whether the nets are shared across time bins

@dataclass
class ConvStackParams:
    hidden_channels: list[int] | None = None
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    dropout: float = 0.0
    batch_norm: bool = True
    zero_init_final: bool = True        # zero weights and bias of the final conv, so that the residual updates of the HLPD block start at zero

@dataclass
class RayTrafoParams:
    geometry: str = "ConeBeamGeometry"
    nDetectorCols: int = 512
    nDetectorRows: int = 96
    DetectorRowExtent: float = 80.0  # in mm
    DetectorColExtent: float = 908.8  # in mm
    GantrySpeed: float = 2 * torch.pi  # in radians per time unit
    nViews: int = 512
    Flux: float = 10e14
    source_radius: float = 540
    det_radius: float = 950
    det_curvature_radius: tuple[float, float] = (950, torch.inf)
    pitch: float = 0.0
    rotAxis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # Default rotation axis (z-axis)
    shape: None | tuple[int, ...] = None
    extent: None | tuple[float, ...] = None

@dataclass
class NormalizationParams:
    operator: bool = False      # Use T_t / ||T_t|| in the HLPD iterations and feed them g_t / ||T_t||, see NormalizedRayTransform
    power_iters: int = 20       # Maximum number of power iterations to estimate ||T_t||, once per geometry
    intensity: str = "physical" # "physical": the iterations work on attenuation values in mm^-1 (about 0.02 for water).
                                # "water": in units of the attenuation of water, so that water is 1 and the nets work on O(1) values.
                                # The data is scaled by the same factor (T is linear), and the output is in mm^-1 in both modes.

    def __post_init__(self) -> None:
        assert self.intensity in ("physical", "water"), f"Invalid intensity mode {self.intensity}. Expected 'physical' or 'water'."

@dataclass
class DataParams:
    shape: tuple[int, ...]
    extent: tuple[float, ...]
    time_steps: list[float] 


@dataclass
class DeformParams:
    N: int = 7  # Default number of integration steps for the velocity field
    extent: tuple[float, ...] | None = None  # Physical extent, set from the scan geometry in HLPDModel.set_geometry
    action: str = "geometric"
    integration: str = "euler"



class ConvBlock(nn.Module):
    """This class implements a simple convolutional block, which consists of a convolutional layer followed by a ReLU activation function. 
    This block can be used as a building block for the CNN modules in the HLPD model."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True, final: bool = False)->None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        # NOTE: in place, as the conv's backward does not need its output. This saves one activation of out_channels per block.
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True) if not final else nn.Identity() # We use LeakyReLU as the activation function for all blocks except the final block, where we use an identity function to allow for unbounded output values. This can be customized based on the requirements of the project (e.g., using different activation functions, etc.).
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 and not final else nn.Identity()
        # self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=out_channels) if batch_norm and not final else nn.Identity() # GroupNorm can be used as an alternative to BatchNorm, especially for small batch sizes. Here we use 4 groups (so out_channels must be divisible by 4), but this can be customized based on the requirements of the project (e.g., using different numbers of groups, etc.).

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """Apply the convolutional block to the input tensor."""

        
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.group_norm(x)
        return x

class ConvStack(nn.Module):
    """This class implements a simple CNN module, which consists of a sequence of convolutional blocks with specified channels, kernel size, padding, stride, dropout, and batch normalization. 
    This module can be used as a building block for the HLPD model, and can be customized based on the requirements of the project (e.g., using different numbers of convolutional blocks, etc.)."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hidden_channels: list[int] | None, 
                 kernel_size: int = 3, 
                 padding: int = 1, 
                 stride: int = 1, 
                 dropout: float = 0.0, 
                 batch_norm: bool = True,
                 zero_init_final: bool = True)->None:
        super().__init__()
        self.blocks = nn.ModuleList()
        if hidden_channels is None:
            hidden_channels = []
        channels = [in_channels] + hidden_channels + [out_channels]
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], kernel_size, padding, stride, dropout, batch_norm, final=True if i == len(channels) - 2 else False)) # We set final=True for the last block to allow for unbounded output values, which can be useful for regression tasks. This can be customized based on the requirements of the project (e.g., using different activation functions for the final block, etc.).
        if zero_init_final:
            # NOTE: the HLPD block adds the output to the previous iterate, so a zero final layer makes the untrained model return its
            # initialization (e.g. the FBP). The other layers still get gradients once the final layer has moved away from zero.
            final_conv = cast(ConvBlock, self.blocks[-1]).conv
            nn.init.zeros_(final_conv.weight)
            assert final_conv.bias is not None
            nn.init.zeros_(final_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the CNN module to the input tensor."""
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor. Received input of type: " + type(x).__name__

        for block in self.blocks:
            x = block(x)
        return x





class NormalizedRayTransform:
    """The ray transform as seen by the HLPD iterations: the operator of time bin t scaled by scales[t].

    With scales[t] = 1 / ||T_t|| (NormalizationParams.operator), T_t^* T_t has norm 1, so that the dual and primal nets get T f and
    T^* h on the scale of f and h rather than ||T_t|| and ||T_t||^2 times larger. The data must then be scaled by the same factor,
    which keeps it consistent with the normalized operator (T_t x = g_t <=> s_t T_t x = s_t g_t), see HLPDModel.scale_data.
    The scales are set per geometry by HLPDModel.set_geometry, and are all 1 (the plain ray transform) without normalization."""

    def __init__(self, ray_trafo: RayTransform, T: int) -> None:
        self.ray_trafo = ray_trafo
        self.scales = [1.0] * T

    @staticmethod
    def _scale(x: torch.Tensor, s: float) -> torch.Tensor:
        return x if s == 1.0 else x * s

    def forward_bin(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self._scale(self.ray_trafo.forward_bin(x, t), self.scales[t])

    def adjoint_bin(self, y: torch.Tensor, t: int) -> torch.Tensor:
        return self._scale(self.ray_trafo.adjoint_bin(y, t), self.scales[t])

    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.forward_bin(xt, t) for t, xt in enumerate(x)]

    def adjoint(self, y: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.adjoint_bin(yt, t) for t, yt in enumerate(y)]


class HLPDBlock(nn.Module):
    """Iteration model for the HLPD framework.

    This class defines a single iteration of the HLPD model, including the primal, dual,
    and registration modules, as well as the ray transform module.
    """
    def __init__(
            self,
            primal_net: ConvStack,              # Primal module
            dual_net: ConvStack,                # Dual module
            motion_net: ConvStack,              # Registration module
            ray_trafo: RayTransform | NormalizedRayTransform,  # Ray transform module (analytical)
            recurrent: bool,                    # Whether the model is recurrent in time
            time_steps: int,
            shared: bool,                       # Recurrent mode only: whether the nets are shared across time bins
            ) -> None:

        super().__init__()

        self.time_steps = time_steps
        self.recurrent = recurrent
        self.shared = shared

        # Define blocks

        self.ray_trafo = ray_trafo

        self.dual_net = self._init_module(dual_net, shared, recurrent)
        self.primal_net = self._init_module(primal_net, shared, recurrent)
        self.motion_net = self._init_module(motion_net, shared, recurrent)


        
        # Define the layers of the iteration model here

    @overload
    def _init_module(self, module: ConvStack, shared: bool, recurrent: Literal[False]) -> nn.Module: ...

    @overload
    def _init_module(self, module: ConvStack, shared: bool, recurrent: Literal[True]) -> nn.ModuleList: ...


    def _init_module(self, module: ConvStack, shared: bool, recurrent: bool=False) -> nn.Module | nn.ModuleList: 

        if not recurrent:
            return module
        else:
            return nn.ModuleList([copy.deepcopy(module) for _ in range(self.time_steps)]) if not shared else nn.ModuleList([module])

    def _time_step(self, t: int, f_prev: torch.Tensor, h_prev: torch.Tensor, v_prev: torch.Tensor, f_t: torch.Tensor, h_t: torch.Tensor,
                   v_t: torch.Tensor, g_t: torch.Tensor, zero_primal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent mode: the update of time bin t, from its iterates (f_prev, h_prev, v_prev) of the previous iteration and the
        iterates (f_t, h_t, v_t) of the previous time bin of the current iteration. Returns the iterates of time bin t."""
        assert isinstance(self.dual_net, nn.ModuleList) and isinstance(self.primal_net, nn.ModuleList) and isinstance(self.motion_net, nn.ModuleList)

        # Update the primal, dual, and registration variables for the current iteration
        # NOTE: the region labels show up under torch.profiler and are sampled by a recording tracking.timeline.StepTimeline, and are no-ops otherwise.

        with region("hlpd/ray_trafo"):
            Tf_t = torch.zeros_like(g_t) if zero_primal else self.ray_trafo.forward_bin(f_prev, t)    # forward projection of the same time step's primal variable from the previous iteration

        idx = 0 if self.shared else t
        gamma_module = self.dual_net[idx]
        lambda_module = self.primal_net[idx]
        sigma_module = self.motion_net[idx]

        # NOTE: the updates are residual on the same time bin's iterate of the previous iteration (h_prev, f_prev, v_prev). The previous
        # time bin's iterate (h_t, f_t, v_t) is only an input, as residuals on it would accumulate over all time bins and iterations.
        with region("hlpd/dual_net"):
            h_t = h_prev + gamma_module(
                torch.stack([ # Stack the relevant tensors along the channel dimension
                    h_prev,         # same time, previous iteration
                    h_t,            # previous time, current iteration
                    Tf_t,           # forward projection of the same time step, previous iteration
                    g_t             # observed data for the current time step
                ], dim=1)
                ).squeeze(1)

        with region("hlpd/ray_trafo_adjoint"):
            Tadj_curr_t = self.ray_trafo.adjoint_bin(h_t, t)

        with region("hlpd/primal_net"):
            f_t = f_prev + lambda_module(
                torch.cat([
                    f_prev.unsqueeze(1),         # same time, previous iteration
                    f_t.unsqueeze(1),            # previous time, current iteration
                    Tadj_curr_t.unsqueeze(1),    # adjoint of the current time step's dual variable
                    v_t                          # previous time, current iteration
                ], dim=1)
                ).squeeze(1)

        with region("hlpd/motion_net"):
            v_t = v_prev + sigma_module(
                torch.cat([
                    v_prev,                      # same time, previous iteration
                    v_t,                         # previous time, current iteration
                    f_t.unsqueeze(1)             # current time, updated primal variable
                ], dim=1)
                )

        return f_t, h_t, v_t

    def _labeled_time_step(self, label: str, *args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: the region is inside the checkpointed function, so that its recompute during backward is labeled as well (see tracking/timeline.py).
        with region(label):
            return self._time_step(*args)

    @overload
    def forward(self, x: Tuple[List[torch.Tensor], ...], g: List[torch.Tensor], zero_primal: bool = False, checkpoint_label: str | None = None) -> Tuple[List[torch.Tensor], ...]: ...
    @overload
    def forward(self, x: Tuple[torch.Tensor, ...], g: torch.Tensor, zero_primal: bool = False, checkpoint_label: str | None = None) -> Tuple[torch.Tensor, ...]: ...

    def forward(self, x: Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...], g: List[torch.Tensor] | torch.Tensor, zero_primal: bool = False, checkpoint_label: str | None = None) -> Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...]:
        """Forward pass of the HLPD iteration model.

        Args:
            x: A tuple containing three elements (f, h, v), each of which can be
                either a list of torch.Tensor or a single torch.Tensor. Where
                f represents the primal variable, 
                h represents the dual variable,
                v represents the registration variable,
            g: The observed data, either as a list of torch.Tensor or a single torch.Tensor.
            zero_primal: Whether the primal variable f is known to be zero (the first iteration of a zero initialized model), in which
                case its forward projection is zero as well and is not computed.
            checkpoint_label: Recurrent mode only: checkpoint each time step on its own, in a region labeled <checkpoint_label>/t<t>.

        Returns:
            A tuple containing the updated (f, h, v) after one iteration of the model.
        """


        _validate_input(x)

        f, h, v = x

        assert isinstance(f, list) and isinstance(h, list) and isinstance(v, list) and isinstance(g, list), f"Currently, only input as lists of torch.Tensor are supported."
        if self.recurrent:

            assert isinstance(self.dual_net, nn.ModuleList)
            assert isinstance(self.primal_net, nn.ModuleList)
            assert isinstance(self.motion_net, nn.ModuleList)


            # Initialize the previous time states

            f_t = f[-1] # We initialize t=0 primal variable with the last time step from the previous iteration
            h_t = h[-1] # We initialize t=0 dual variable with the last time step from the previous iteration
            v_t = v[-1] # We initialize t=0 registration variable with the last time step from the previous iteration



            f_pred = []
            h_pred = []
            v_pred = []

            for t in range(self.time_steps):
                # NOTE: with checkpoint_label, each time step is checkpointed on its own (see HLPDUnrolled). Its inputs are kept anyway, as
                # they are iterates or data, so only the activations of a single time step are held during the backward pass.
                if checkpoint_label is not None:
                    f_t, h_t, v_t = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], checkpoint(
                        self._labeled_time_step, f"{checkpoint_label}/t{t}", t, f[t], h[t], v[t], f_t, h_t, v_t, g[t], zero_primal, use_reentrant=False))
                else:
                    f_t, h_t, v_t = self._time_step(t, f[t], h[t], v[t], f_t, h_t, v_t, g[t], zero_primal)

                f_pred.append(f_t)
                h_pred.append(h_t)
                v_pred.append(v_t)

        else:
            assert isinstance(self.dual_net, ConvStack)
            assert isinstance(self.primal_net, ConvStack)
            assert isinstance(self.motion_net, ConvStack)

            with region("hlpd/ray_trafo"):
                Tf_s = torch.zeros_like(torch.stack(g, dim=1)) if zero_primal else torch.stack(self.ray_trafo(f), dim=1)
            f_s = torch.stack(f, dim=1)
            h_s = torch.stack(h, dim=1)
            v_s = torch.cat(v, dim=1)             # (B, 3T, D, H, W) with channel order [v0x, v0y, v0z, v1x, ...]
            g_s = torch.stack(g, dim=1)

            # NOTE: the updates are residual on the previous iteration, as in the recurrent case.
            with region("hlpd/dual_net"):
                h_pred: list[torch.Tensor] = list((h_s + self.dual_net(
                    torch.cat([
                            h_s,         # same time, previous iteration
                            Tf_s,        # forward projection of the primal variable from the previous iteration
                            g_s          # observed data
                        ], dim=1))).unbind(dim=1))

            with region("hlpd/ray_trafo_adjoint"):
                Tadjh_s = torch.stack(self.ray_trafo.adjoint(h_pred), dim=1)

            with region("hlpd/primal_net"):
                f_pred: list[torch.Tensor] = list((f_s + self.primal_net(
                    torch.cat([
                        f_s,         # same time, previous iteration
                        Tadjh_s,     # adjoint of the updated dual variable
                        v_s          # previous iteration
                    ], dim=1))).unbind(dim=1))

            with region("hlpd/motion_net"):
                v_pred: list[torch.Tensor] = list((v_s + self.motion_net(
                    torch.cat([
                        v_s,                            # previous iteration
                        torch.stack(f_pred, dim=1)      # updated primal variable
                    ], dim=1)
                )).split(3, dim=1))                     # T tensors of shape (B, 3, D, H, W)


        # Define the forward pass of the iteration model here
        return (f_pred, h_pred, v_pred)




class HLPDUnrolled(nn.Module):
    """Hierarchical Learned Primal-Dual (HLPD) Model.

    This class implements the HLPD framework for iterative reconstruction and registration.
    It applies a learned iteration model multiple times, optionally sharing the model across iterations.
    """
    def __init__(
            self,
            hlpd_iter_model: HLPDBlock,         # The model used for each iteration of the HLPD process. Maps the triple $(x^{(k)}, y^{(k)}, v^{(k)}) \mapsto (x^{(k+1)}, y^{(k+1)}, v^{(k+1)})$
            num_iters: int,                     # The number of iterations to apply the HLPD iteration model.
            shared: bool = False,               # Whether the HLPD iteration model is shared across iterations.
            grad_checkpointing: bool = False,  # Whether to use gradient checkpointing for the HLPD iteration model.
            ) -> None:
        super(HLPDUnrolled, self).__init__()
        # Initialize model components here
        self.num_iters = num_iters
        self.shared = shared
        self.grad_checkpointing = grad_checkpointing

        # Initialize the module list for the iteration models
        # NOTE: hlpd_iter_model is not registered as a submodule itself. When not shared it is only a template for the copies, and registering it would add unused parameters.

        if self.shared:
            self.hlpd_model_list = nn.ModuleList([hlpd_iter_model] * self.num_iters)
        else:
            # NOTE: the ray transform is an analytical operator whose geometry is updated in place (HLPDModel.set_geometry), so it must be shared by all iterations rather than copied.
            ray_trafo = hlpd_iter_model.ray_trafo
            self.hlpd_model_list = nn.ModuleList([copy.deepcopy(hlpd_iter_model, memo={id(ray_trafo): ray_trafo}) for _ in range(self.num_iters)])

    @overload
    def forward(self, x: Tuple[List[torch.Tensor], ...], g: List[torch.Tensor], zero_init: bool = False) -> Tuple[List[torch.Tensor], ...]: ...
    
    @overload
    def forward(self, x: Tuple[torch.Tensor, ...], g: torch.Tensor, zero_init: bool = False) -> Tuple[torch.Tensor, ...]: ...

    def forward(self, x: Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...], g: List[torch.Tensor] | torch.Tensor, zero_init: bool = False) -> Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...]:
        """Forward pass of the HLPD unrolled model.

        Args:
            x: The input tuple of primal, dual and velocity variables. These are the initialized values for the HLPD iterations (default is zero initialization).
            g: The raw data (e.g., measurements from the imaging system).
            zero_init: Whether the primal variable of x is zero, see HLPDBlock.forward.

        Returns:
            The updated tuple of primal and dual variables after applying the HLPD iterations.
        """
        # Define the forward pass here

        # validate the input

        _validate_input(x)

        
        for k, hlpd_iter in enumerate(self.hlpd_model_list, start=1):
            label = f"hlpd/iter{k}"
            # NOTE: the region is inside the checkpointed function, so that its recompute during backward is labeled as well (see tracking/timeline.py).
            def run_iter(x, g, hlpd_iter=hlpd_iter, label=label, zero_primal=zero_init and k == 1):
                with region(label):
                    return hlpd_iter(x, g, zero_primal=zero_primal)

            if self.grad_checkpointing and hlpd_iter.recurrent:
                # NOTE: the recurrent block checkpoints each time step instead of the whole iteration. That recomputes the same, but only
                # holds the activations of one time step rather than of all T at a time during the backward pass (about 1.5 GB each at
                # full resolution).
                with region(label):
                    x = hlpd_iter(x, g, zero_primal=zero_init and k == 1, checkpoint_label=label)
            elif self.grad_checkpointing:
                x = cast(
                    Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...],
                    checkpoint(run_iter, x, g, use_reentrant=False),
                )
            else:
                x = run_iter(x, g)
        return x






class HLPDModel(nn.Module):
    """Hierarchical Learned Primal-Dual (HLPD) Model.

    This class implements the full HLPD model, which consists of multiple iterations of the HLPD iteration model.
    It can optionally share the iteration model across iterations and use gradient checkpointing for memory efficiency.
    """
    def __init__(self, 
                unrolled_params: HLPDUnrolledParams, 
                block_params: HLPDBlockParams, 
                conv_stack_params: ConvStackParams,
                ray_trafo_params: RayTrafoParams, 
                deform_params: DeformParams,
                T: int, # Number of time bins
                shape: tuple[int, int, int], 
                zero_init: bool = True,
                normalization: NormalizationParams | None = None,
                mu_water: float | None = None,  # attenuation of water in mm^-1, needed for the "water" intensity mode
                device: torch.device = torch.device("cpu"))-> None:
        
        super(HLPDModel, self).__init__()

        # architecture parameters

        self.unrolled_params = unrolled_params
        self.block_params = block_params
        self.conv_stack_params = conv_stack_params

        # operator parameters

        self.ray_trafo_params = ray_trafo_params
        self.deform_params = deform_params
        self.zero_init = zero_init
        self.normalization = normalization if normalization is not None else NormalizationParams()
        if self.normalization.intensity == "water":
            assert mu_water is not None and mu_water > 0, "The 'water' intensity mode needs the attenuation of water mu_water."
        self.intensity_scale = 1.0 / mu_water if self.normalization.intensity == "water" and mu_water is not None else 1.0  # images in the iterations are intensity_scale * x

        self.device = device

        # set spatio-temporal shape params --> general shape is T x B x *spatial_dims where the first (time) dimension is represented with a list

        self.T = T          # time bins

        self.shape = shape  # spatial shape
        self.sino_shape = (ray_trafo_params.nViews, ray_trafo_params.nDetectorCols, ray_trafo_params.nDetectorRows) # projection shape, odl orders the projection space as (angles, *detector)

        ## ----------------- Build operators

        self.ray_trafo = RayTransform(**asdict(ray_trafo_params))                     # the physical operator, also used by the projection domain loss
        self.block_ray_trafo = NormalizedRayTransform(self.ray_trafo, T)                # the (normalized) operator used in the HLPD iterations
        self._operator_norms: dict[tuple, list[float]] = {}                             # ||T_t|| per geometry, see set_geometry
        self.deform = FlowDeformationOperator(**asdict(deform_params), shape=shape, device=device)

        # Build the primal, dual and sigma nets

        recurrent = block_params.recurrent

        p_in = 6 if recurrent else 5 * self.T
        p_out = 1 if recurrent else self.T

        d_in = 4 if recurrent else 3 * self.T
        d_out = 1 if recurrent else self.T

        m_in = 7 if recurrent else 4 * self.T
        m_out = 3 if recurrent else 3 * self.T

        # NOTE: the nets and the block are kept as locals so that the only registered copies of the parameters are the ones in self.unrolled_model.
        primal_net = ConvStack(in_channels=p_in, out_channels=p_out, **asdict(conv_stack_params))
        dual_net = ConvStack(in_channels=d_in, out_channels=d_out, **asdict(conv_stack_params))
        sigma_net = ConvStack(in_channels=m_in, out_channels=m_out, **asdict(conv_stack_params))

        ## ----------------- Build HLPD block

        hlpd_block = HLPDBlock(primal_net, dual_net, sigma_net, self.block_ray_trafo, **asdict(block_params))

        ## ----------------- Initialize the unrolled HLPD model

        self.unrolled_model = HLPDUnrolled(hlpd_iter_model=hlpd_block, **asdict(unrolled_params)).to(device)



    def set_geometry(self, meta: dict):
        """We have to set the spatial geometry parameters dynamically for each volume since the acqusition protocols differed for the scans.
        This mainly affects the number of slices on the slice spacing. This means that we also need to redefine the ray transform dynamically 
        since it depends on the 3D reconstruction space
        Args:
            meta: dict containing the scan specific 
        """
        extent = study_extent(meta, self.shape)
        self.extent = extent
        time_steps = [float(dt) for dt in meta.get("time_steps", [1.0] * self.T)] # the time steps the data was simulated with, see data/pipeline.py
        assert len(time_steps) == self.T, f"Expected {self.T} time bins, but the study has {len(time_steps)}."
        self.ray_trafo._init_ray_transform(self.shape, extent, time_steps)
        self.deform.extent = extent # the deformation operator normalizes velocities with the physical extent

        if self.normalization.operator:
            key = (tuple(self.shape), tuple(float(e) for e in extent), tuple(time_steps))
            if key not in self._operator_norms:
                self._operator_norms[key] = [
                    operator_norm(lambda x, t=t: self.ray_trafo.forward_bin(x, t), lambda y, t=t: self.ray_trafo.adjoint_bin(y, t),
                                  (1, *self.shape), max_iters=self.normalization.power_iters, device=self.device)
                    for t in range(self.T)
                ]
            self.operator_norms = self._operator_norms[key]
            self.block_ray_trafo.scales = [1.0 / n for n in self.operator_norms]

    def scale_data(self, g: list[torch.Tensor]) -> list[torch.Tensor]:
        """The data g_t (in intensity units, see NormalizationParams.intensity) on the scale of the operator used in the HLPD iterations,
        i.e. g_t / ||T_t|| when normalizing the operator (see NormalizedRayTransform)."""
        return [NormalizedRayTransform._scale(gt, s) for gt, s in zip(g, self.block_ray_trafo.scales)]

    def initialize(self, g: list[torch.Tensor]) -> Tuple[List[torch.Tensor], ...]:
        """Zero initialization of (f, h, v), with batch size, device and dtype taken from the data g."""

        B = g[0].shape[0]
        kwargs = {"device": g[0].device, "dtype": g[0].dtype}


        v0 = [torch.zeros((B, 3, *self.shape), **kwargs) for t in range(self.T)] # NOTE internally, it is convenient to use (T, B, 3, D, H, W) for v, but the flow deform operator expects (T, B, D, H, W, 3) so we will have to transform the output to match this format

        if self.zero_init:
            f0 = [torch.zeros((B, *self.shape), **kwargs) for t in range(self.T)]
            h0 = [torch.zeros((B, *self.sino_shape), **kwargs) for t in range(self.T)]
        else:
            # FBP initilization 
            f0 = self.ray_trafo.FBP(g)   # NOTE: FBP inverts the physical operator, so it gets the unscaled data
            h0 = self.scale_data(g)
        
        return (f0, h0, v0)


    def forward(self, g: list[torch.Tensor]) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Reconstructs the template and the time dependent velocity fields from the data g (list of T tensors B x nAngles x nCol x nRow).

        Returns:
            f_template: The reconstructed template B x D x H x W (the primal variable of the first time bin).
            v: List of T velocity fields B x D x H x W x 3, as expected by self.deform. The deformed sequence is self.deform(f_template, v),
               which the LDDMM loss computes itself, so it is not computed here.
        """

        assert hasattr(self.ray_trafo, "ray_transform"), f"Must initialize the ray_trafo operator before calling the forward method. This is done with self.set_geometry"

        # NOTE: in general we will want to update the ray transform to match the geometry of the input. Make sure to do before calling the forward method. 

        # NOTE: the iterations work on images in intensity units (see NormalizationParams.intensity), and on data scaled accordingly, which
        # is consistent as T is linear: T(c x) = c g. The FBP initialization in self.initialize is then in intensity units as well.
        c = self.intensity_scale
        g = [gt * c for gt in g] if c != 1.0 else g
        x = self.initialize(g)

        with region("hlpd/unrolled"):
            out: Tuple[List[torch.Tensor], ...] = self.unrolled_model(x, self.scale_data(g), zero_init=self.zero_init)
        f, h, v = out

        v = [vt.permute(0, 2, 3, 4, 1) for vt in v] # NOTE the flow deformation operator expects List[B, D, H, W, 3]
        f_template = f[0] / c if c != 1.0 else f[0]  # back to mm^-1

        return (f_template, v)
