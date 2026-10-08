from typing import Tuple, List, overload, Literal, cast
from dataclasses import dataclass, asdict
# torch imports

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import copy


## --------------- LOCAL IMPORTS ---------------

from operators.tomo.ray_trafo import RayTransform
from operators.lddmm.deform import FlowDeformationOperator
from data.utils import study_extent

def _validate_input(x: Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...]) -> None:
    if not isinstance(x, tuple) or len(x) != 4:
        raise ValueError("Input must be a tuple of at least four elements.")
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
    shared: bool = False                                                        # Whether the model is shared across iterations

@dataclass
class ConvStackParams:
    hidden_channels: list[int] | None = None
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    dropout: float = 0.0
    batch_norm: bool = True

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

@dataclass
class DataParams:
    shape: tuple[int, ...]
    extent: tuple[float, ...]
    time_steps: list[float]


@dataclass
class DeformParams:
    N: int = 7  # Default number of integration steps for the velocity field
    extent: tuple[float, float] = (450.0, 450.0)
    action: str = "geometric"
    integration: str = "euler"



class ConvBlock(nn.Module):
    """This class implements a simple convolutional block, which consists of a convolutional layer followed by a ReLU activation function. 
    This block can be used as a building block for the CNN modules in the HLPD model."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True, final: bool = False)->None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.activation = nn.LeakyReLU(negative_slope=0.01) if not final else nn.Identity() # We use LeakyReLU as the activation function for all blocks except the final block, where we use an identity function to allow for unbounded output values. This can be customized based on the requirements of the project (e.g., using different activation functions, etc.).
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 and not final else nn.Identity()
        # self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=out_channels) if batch_norm and not final else nn.Identity() # GroupNorm can be used as an alternative to BatchNorm, especially for small batch sizes. Here we use 8 groups, but this can be customized based on the requirements of the project (e.g., using different numbers of groups, etc.).

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
                 batch_norm: bool = True)->None:
        super().__init__()
        self.blocks = nn.ModuleList()
        if hidden_channels is None:
            hidden_channels = []
        channels = [in_channels] + hidden_channels + [out_channels]
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], kernel_size, padding, stride, dropout, batch_norm, final=True if i == len(channels) - 2 else False)) # We set final=True for the last block to allow for unbounded output values, which can be useful for regression tasks. This can be customized based on the requirements of the project (e.g., using different activation functions for the final block, etc.).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the CNN module to the input tensor."""
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor. Received input of type: " + type(x).__name__

        for block in self.blocks:
            x = block(x)
        return x





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
            ray_trafo_module: RayTransform,      # Ray transform module (analytical)
            recurrent: bool,                    # Whether the model is recurrent in time
            time_steps: int,
            shared: bool,                       # Whether the model is shared across iterations
            ) -> None:

        super().__init__()

        self.time_steps = time_steps
        self.recurrent = recurrent
        self.shared = shared

        # Define blocks

        self.ray_trafo = ray_trafo_module

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

    @overload
    def forward(self, x: Tuple[List[torch.Tensor], ...], g: List[torch.Tensor]) -> Tuple[List[torch.Tensor], ...]: ...
    @overload
    def forward(self, x: Tuple[torch.Tensor, ...], g: torch.Tensor) -> Tuple[torch.Tensor, ...]: ...

    def forward(self, x: Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...], g: List[torch.Tensor] | torch.Tensor) -> Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...]:
        """Forward pass of the HLPD iteration model.

        Args:
            x: A tuple containing three elements (f, h, v), each of which can be
                either a list of torch.Tensor or a single torch.Tensor. Where
                f represents the primal variable, 
                h represents the dual variable,
                v represents the registration variable,
            g: The observed data, either as a list of torch.Tensor or a single torch.Tensor.

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


                # set time bin for the ray transform

                self.ray_trafo._set_time_bin(t)
                
                # Update the primal, dual, and registration variables for the current iteration 

                Tf_t = self.ray_trafo(f_t)



                gamma_module = self.dual_net[t]
                lambda_module = self.primal_net[t]
                sigma_module = self.motion_net[t]

                h_t = gamma_module(
                    torch.stack([ # Stack the relevant tensors along the channel dimension
                        
                        h[t],           # same time, previous iteration
                        h_t,            # previous time, current iteration
                        Tf_t,           # forward projection of the previous time step's primal variable
                        g[t]            # observed data for the current time step
                    ], dim=1)
                    )

                Tadj_curr_t = self.ray_trafo.adjoint(h_t)


                f_t = lambda_module(
                    torch.stack([
                        f[t],           # same time, previous iteration
                        f_t,            # previous time, current iteration
                        Tadj_curr_t,    # adjoint of the current time step's dual variable
                        v_t             # previous time, current iteration
                    ], dim=1)
                    )

                v_t = sigma_module(
                    torch.stack([
                        v[t],           # same time, previous iteration
                        v_t,            # previous time, current iteration
                        f_t             # current time, updated primal variable
                    ], dim=1)
                    )



                f_pred.append(f_t)
                h_pred.append(h_t)
                v_pred.append(v_t)


    
        else:
            assert isinstance(self.dual_net, ConvStack)
            assert isinstance(self.primal_net, ConvStack)
            assert isinstance(self.motion_net, ConvStack)


        

            Tf = torch.stack(self.ray_trafo(f), dim=1)
            f = torch.stack(f, dim=1)
            h = torch.stack(h, dim=1)
            v = torch.stack(v, dim=1)
            g = torch.stack(g, dim=1)


    

            h_pred: list[torch.Tensor] = list(self.dual_net(
                torch.cat([
                        h,           # same time, previous iteration
                        Tf,          # forward projection of the previous time step's primal variable
                        g            # observed data for the current time step
                    ], dim=1)).unbind(dim=1))
            Tadjh = torch.stack(self.ray_trafo.adjoint(h_pred), dim=1)


            f_pred: list[torch.Tensor] = list(self.primal_net(
                torch.cat([
                    f,           # same time, previous iteration
                    Tadjh,    # adjoint of the current time step's dual variable
                    v            # previous time, current iteration
                ], dim=1)).unbind(dim=1))


            f = torch.stack(f_pred, dim=1)
            v_pred: list[torch.Tensor] = list(self.motion_net(
                torch.cat([
                    v,           # same time, previous iteration
                    f            # current time, updated primal variable
                ], dim=1)
            ).unbind(dim=1))


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
        self.hlpd_iter_model = hlpd_iter_model
        self.num_iters = num_iters
        self.shared = shared
        self.grad_checkpointing = grad_checkpointing

        # Initialize the module list for the iteration models

        if self.shared:
            self.hlpd_model_list = nn.ModuleList([self.hlpd_iter_model] * self.num_iters)
        else:
            self.hlpd_model_list = nn.ModuleList([copy.deepcopy(self.hlpd_iter_model) for _ in range(self.num_iters)])

    @overload
    def forward(self, x: Tuple[List[torch.Tensor], ...], g: List[torch.Tensor]) -> Tuple[List[torch.Tensor], ...]: ...
    
    @overload
    def forward(self, x: Tuple[torch.Tensor, ...], g: torch.Tensor) -> Tuple[torch.Tensor, ...]: ...

    def forward(self, x: Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...], g: List[torch.Tensor] | torch.Tensor) -> Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...]:
        """Forward pass of the HLPD unrolled model.

        Args:
            x: The input tuple of primal, dual and velocity variables.
            g: The raw data (e.g., measurements from the imaging system).

        Returns:
            The updated tuple of primal and dual variables after applying the HLPD iterations.
        """
        # Define the forward pass here

        # validate the input

        _validate_input(x)

        
        for hlpd_iter in self.hlpd_model_list:
            if self.grad_checkpointing:
                x = cast(
            Tuple[List[torch.Tensor], ...] | Tuple[torch.Tensor, ...],
            checkpoint(hlpd_iter, x, use_reentrant=False),
        )
            else:
                x = hlpd_iter(x)
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
                device: torch.device = torch.device("cpu"))-> None:
        
        super(HLPDModel, self).__init__()

        # architecture parameters

        self.unrolled_params = unrolled_params
        self.block_params = block_params
        self.conv_stack_params = conv_stack_params

        # operator parameters

        self.ray_trafo_params = ray_trafo_params
        self.deform_params = deform_params


        # 

        self.device = device
        self.T = T


        ## ----------------- Build operators

        self.ray_trafo = RayTransform(**asdict(ray_trafo_params))
        self.deform = FlowDeformationOperator(**asdict(deform_params))

        # Build the primal, dual and sigma nets

        recurrent = block_params.recurrent

        p_in = 6 if recurrent else 5 * self.T
        p_out = 1 if recurrent else self.T

        d_in = 4 if recurrent else 3 * self.T
        d_out = 1 if recurrent else self.T

        m_in = 7 if recurrent else 4 * self.T
        m_out = 3 if recurrent else 3 * self.T

        self.primal_net = ConvStack(in_channels=p_in, out_channels=p_out, **asdict(conv_stack_params)).to(device)
        self.dual_net = ConvStack(in_channels=d_in, out_channels=d_out, **asdict(conv_stack_params)).to(device)
        self.sigma_net = ConvStack(in_channels=m_in, out_channels=m_out, **asdict(conv_stack_params)).to(device)

        ## ----------------- Build HLPD block

        self.hlpd_block = HLPDBlock(self.primal_net, self.dual_net, self.sigma_net, self.ray_trafo, **asdict(block_params)).to(device)

        ## ----------------- Initialize the unrolled HLPD model

     
        # Initialize the unrolled HLPD model
        self.unrolled_model = HLPDUnrolled(hlpd_iter_model=self.hlpd_block, **asdict(unrolled_params)).to(device)



    def update_ray_trafo(self, meta, shape):
        extent = study_extent(meta, shape)
        time_steps = [1.0] * self.T
        self.ray_trafo._init_ray_transform(shape, extent, time_steps)


    def forward(self, g: list[torch.Tensor], meta, shape):

        # NOTE: in general we will want to update the ray transform to match the geometry of the input
        self.update_ray_trafo(meta, shape)

        return self.unrolled_model(g)