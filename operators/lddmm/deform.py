from os import stat

import torch
from torch.nn.functional import grid_sample, pad
from typing import overload, Tuple
from itertools import chain


class GroupAction(torch.nn.Module):
    r"""This class implements the group action (currently only the geometric group action) of diffeomorphisms on images. 
    The group action is defined as $\\mathcal{V}_{\phi} x = x \circ \phi^{-1}$, where $\phi$ is a diffeomorphism and $x$ is an image. 
    The operator takes as input a deformation field and an image, and outputs the deformed image."""
    

    # TODO: implement mass preserving action as well, which is defined as $\\mathcal{V}_{\phi} x = (x \circ \phi^{-1}) |D\phi^{-1}|$, where $|D\phi^{-1}|$ is the Jacobian determinant of the inverse deformation field.

    def __init__(self, action: str = "geometric", device: torch.device = torch.device("cpu")):
        super().__init__()

        # Get identity grid for the given space

        assert action in ("geometric", "mass_preserving"), f"Unsupported action type: {action}. Supported types are 'geometric' and 'mass_preserving'."

        self.action = action
        self.device = device
        self.to(self.device)

    def jacobian_determinant(self, phi: torch.Tensor) -> torch.Tensor:
        r"""This function computes the Jacobian determinant:
        D\phi^{-1} is the Jacobian matrix of the inverse deformation field, and the Jacobian determinant is computed as the determinant of this matrix.
        
        input:
            phi (torch.Tensor): The inverse deformation field of shape (B, *spatial_dims, C), where B is the batch size, *spatial_dims are the spatial dimensions, and C is the number of channels (2 for 2D, 3 for 3D).
            shape (tuple[int, ...]): The spatial dimensions of the image.
            extent (tuple[tuple[float, float], ...] | None): The physical extent of the image domain. If None, unit spacing is assumed.

        returns:
            torch.Tensor: The Jacobian determinant of the inverse deformation field of shape (B, *spatial_dims).
        """

        B = phi.shape[0]
        spatial_dims = phi.shape[1:-1]
        C = phi.shape[-1]

        assert C == len(spatial_dims), f"Channel dimension {C} does not match the length of spatial_dims {len(spatial_dims)}."

        # NOTE: We are estimation the jacobian with central differences. torch.roll uses cyclic boundary conditions, which results incorrect derivatives at the boundaries.
        # For this reason, we pad the input tensor before computing the central differences to avoid incorrect derivatives at the boundaries. Since pad operates on the last dimensions, we need to permute the channel dimension to right after the batch dimension before padding (so the last dimensions really are the spatial ones) and then permute it back afterwards.

        # Pad the input tensor with one voxel on each side along each spatial dimension
        phi = pad(phi.permute(0, -1, *range(1, 1 + len(spatial_dims))), pad=[1, 1] * len(spatial_dims), mode='replicate').permute(0, *range(2, 2 + len(spatial_dims)), 1)

        # Dphi is built from the now-padded phi (each spatial dim is 2 voxels larger than spatial_dims),
        # so it must be allocated at the padded size too -- it gets sliced back down to spatial_dims below.
        Dphi = torch.zeros((B, *phi.shape[1:-1], C, C), device=self.device)

        for i, s in enumerate(spatial_dims):


            # Estimate the derivative with a difference quotient.
            # NOTE: We need to deal with the normalization.
            # phi is assumed to be in the range [-1, 1], so we first denormalize it to the actual physical spacing by multiplying by (e[1]-e[0])/2.
            # Then we divide by the spacing (e[1]-e[0])/(s-1) -- s-1, not s, since a length-s linspace over [-1,1] has s-1 steps between its
            # endpoints. Equivalently, this simplifies to multiplying by (s-1)/2. We need to divide by an additional factor 2 for the central
            # difference quotient, which yields (s-1)/4. The two roll() calls must be ordered as (forward - backward) = phi[k+1] - phi[k-1] to
            # get the correctly-signed derivative; the previous (backward - forward) ordering computed its negative. In even spatial
            # dimensionality (2D) that sign error cancels in the determinant (both factors flip together), so it can look right in 2D and still
            # be wrong in 3D -- see test/test_jacobian.py's 3D mirror-flip case, which is specifically designed to catch this.
            deriv =  (s-1) / 4 * (phi.roll(shifts=-1, dims=i+1) - phi.roll(shifts=1, dims=i+1))

            # `deriv`'s channel axis is phi's own channel order, i.e. (x, y[, z]) -- required by grid_sample,
            # which is why _identity_grid stacks channels that way (see geometric_action/_compose). Spatial
            # axis `i` here is storage order, i.e. (D,)H,W -- the *reverse* of (x,y[,z]). So this axis's
            # derivative column must be written to the reversed slot C-1-i, not i, or Dphi's rows (channels,
            # x/y/z order) and columns (axes, storage order) end up mismatched: a determinant-sign-flipping
            # permutation for both 2 and 3 channels (reversing 2 or 3 elements is always an odd permutation).
            Dphi[..., C - 1 - i] = deriv

        # Remove the padding from Dphi to match the original spatial dimensions
        Dphi = Dphi[..., 1:-1, 1:-1, 1:-1, :, :] if len(spatial_dims) == 3 else Dphi[..., 1:-1, 1:-1, :, :]

        # Compute the determinant of Dphi:

        detDphi = torch.det(Dphi)

        assert detDphi.shape == (B, *spatial_dims), f"Expected detDphi to have shape {(B, *spatial_dims)}, but got {detDphi.shape}."

        # Compute the Jacobian determinant of the inverse deformation field
        # This is a placeholder implementation and should be replaced with the actual computation
        return detDphi

    def geometric_action(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Apply the group action to the input image using the given velocity field."""

        # Here we would implement the actual deformation logic using the velocity field.
        # This typically involves integrating the velocity field to obtain the deformation and then applying it to the image.

        # TODO: This implementation is basedon pytorch grid_sample function which uses bilinears interpolation. 
        # However, we are working with anisotropic volumes, so perhaps we should implement our own interpolation scheme that takes into account the anisotropy of the volumes. 
        # This is something to consider for future work, but for now we can use the grid_sample function as a starting point.

        # NOTE: It is assumed phi already is normalized to the range [-1, 1] based on the dimensions of the input image. This should be taken care of in the FlowDeformationOperator class, 
        # which normalizes the velocity field before passing it to the GroupAction class.

        deformed = grid_sample(x, phi, align_corners=True, mode="bilinear", padding_mode="zeros")
 
        return deformed

    def mass_preserving_action(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:


        detDphi = torch.abs(self.jacobian_determinant(phi))
        deformed = self.geometric_action(x, phi) * detDphi

        return deformed

    @overload
    def forward(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor: ...
    @overload
    def forward(self, x: torch.Tensor, phi: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def forward(self, x: torch.Tensor, phi: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:


        assert isinstance(phi, torch.Tensor) or (isinstance(phi, list) and all(isinstance(p, torch.Tensor) for p in phi)), "phi must be a torch.Tensor or a list of torch.Tensors."
        if isinstance(phi, list):
            return [self.forward(x, p) for p in phi]

        # At this point, phi is guaranteed to be a single torch.Tensor.
        
        shape_phi = phi.shape
        C = shape_phi[-1]
        B = shape_phi[0]
        spatial_dims = shape_phi[1:-1]
        x_shape = x.shape

        # Ensure shape compatibility between input image and deformation field

        assert C in (2, 3), f"Expected last dimension of deformation to be 2 for 2D or 3 for 3D, but got {C}."
        assert B == x_shape[0], f"Expected batch size of input image and deformation to be the same, but got {B} and {x_shape[0]}."
        assert x.shape[-len(spatial_dims):] == spatial_dims, f"Expected last {len(spatial_dims)} dimensions of input image to match spatial dimensions of deformation, but got {x.shape[-len(spatial_dims):]} and {spatial_dims}."
        assert x.dim() in (len(spatial_dims) + 1, len(spatial_dims) + 2), f"Expected input image to have {len(spatial_dims) + 1} (without channels) or {len(spatial_dims) + 2} (with channels) dimensions, but got {x.dim()} dimensions with shape {x.shape}."

        # Add channel dimension if the input image does not have one
        if x.dim() == len(spatial_dims) + 1:
            x = x.unsqueeze(1)  # Add channel dimension at position 1

        assert phi.dim() == x.dim(), f"Expected deformation to have the same number of dimensions as input image, but got {phi.dim()} and {x.dim()} dimensions."



        if self.action == "geometric":
            return self.geometric_action(x, phi)
        elif self.action == "mass_preserving":
            return self.mass_preserving_action(x, phi)
        else:
            raise ValueError(f"Unsupported action type: {self.action}")

class VelocityIntegrator(torch.nn.Module):
    """
    This class implements the velocity integration operator, which takes as input a static velocity field and outputs the corresponding deformation field. 
    This is implemented using repeated scaling and squaring, which is a common method for integrating velocity fields in the LDDMM framework. 
    The number of integration steps can be specified as a parameter.
    
    
    input: v (B, D, H, W, 3) for 3D or (B, H, W, 2) for 2D
    output: phi (B, D, H, W, 3) for 3D or (B, H, W, 2) for 2D
    
    
    """
    def __init__(
            self, 
            N: int,                                                 # Number of integration steps for the velocity field
            integration:str="scaling_and_squaring",                 # Integration method for the velocity field
            shape: tuple | None = None,                             # Shape of the input image, used to create the identity grid
            device: torch.device = torch.device("cpu")              # Device on which computations will be performed
            )->None:
        super().__init__()

        assert integration in ["scaling_and_squaring", "euler"], f"Unsupported integration method: {integration}. Supported methods are 'scaling_and_squaring' and 'euler'."

        if shape is None:
            print(f"Warning: no value for the shape of the input image was provided for the class {self.__class__.__name__}. The shape is inferred from the input image during the forward pass...\n")
            shape = None  # Default shape of the input image, which is used to create the identity grid for the given space

        self.integration = integration  # Integration method for the velocity field
        self.shape = shape  # Shape of the input image, which is used to create the identity grid for the given space
        self.N = N  # Number of integration steps for the velocity field
        self.device = device  # Device on which the computations will be performed (CPU or GPU)
        self.id = self._identity_grid(shape).to(device) if shape is not None else None  # Identity grid for the given space, which is used for the scaling and squaring method of integration.
        self.to(self.device)

    def _validate_velocity(self, v: torch.Tensor)->None:
        """Validate the velocity field to ensure it has the correct dimensions for 2D or 3D."""
        if v.dim() == 5:
            B, D, H, W, C = v.shape
            assert C == 3, f"Expected velocity field to have 3 channels for 3D, but got {C} channels."
        elif v.dim() == 4:
            B, H, W, C = v.shape
            assert C == 2, f"Expected velocity field to have 2 channels for 2D, but got {C} channels."
        else:
            raise ValueError(f"Unexpected velocity field dimensions: {v.shape}")

    def _init_shape(self, shape: tuple)->None:
        """Initialize the shape of the input image and create the identity grid for the given space."""
        self.shape = shape
        self.id = self._identity_grid(shape).to(self.device)  # Create the identity grid for the given space based on the provided shape. The identity grid is used for the scaling and squaring method of integration.

    def _compose(self, phi:torch.Tensor, psi:torch.Tensor)->torch.Tensor:
        r"""Compose two deformations phi and psi.
        
        input:
            phi (torch.Tensor): The first deformation field.
            psi (torch.Tensor): The second deformation field.

        returns:
            torch.Tensor: The composed deformation field $\$
        
        """
        # Here we would implement the logic for composing two deformations, which is necessary for the LDDMM framework.

        if phi.dim() == 5:
            B, D, H, W, C = phi.shape
            assert C == 3, f"Expected deformation to have 3 channels for 3D deformation, but got {C} channels."
            composed = grid_sample(phi.permute(0, 4, 1, 2, 3), psi, align_corners=True, mode="bilinear", padding_mode="border").permute(0, 2, 3, 4, 1)  # Compute the composition using grid_sample

        else:
            B, H, W, C = phi.shape
            assert C == 2, f"Expected deformation to have 2 channels for 2D deformation, but got {C} channels."
            composed = grid_sample(phi.permute(0, 3, 1, 2), psi, align_corners=True, mode="bilinear", padding_mode="border").permute(0, 2, 3, 1)  # Compute the composition using grid_sample

        return composed
    
    def _square(self, phi: torch.Tensor)->torch.Tensor:
        """Compose the deformations phi with itself"""
        return self._compose(phi, phi)


    def _scale(self, v: torch.Tensor, N: int | None = None) -> torch.Tensor:
        """Scale the velocity field by a given factor."""
        # Here we would implement the logic for scaling the velocity field, which is necessary for the scaling and squaring method of integration.

        if N is None:
            N = self.N

        v_scaled = v / (2 ** N)  # Scale the velocity field by the factor 1/(2^N)

        return v_scaled

    def _identity_grid(self, shape) -> torch.Tensor:
        """
        Here we would implement the logic for creating an identity grid, which is a grid that maps each point to itself. 
        Values must be normalized to the range [-1, 1] for use with PyTorch's grid_sample function. 
        The current implementation only supports 2D and 3D grids.
        """

        assert len(shape) in (2, 3), f"Expected shape to be a 2-tuple (H, W) or 3-tuple (D, H, W), but got {shape}."

        # NOTE: Even if the spatial dimensions use the (D, H, W) or (H, W) convention, corresponding to z, y, x or y, x, the channel order in the identity grid is always (x, y, z) for 3D and (x, y) for 2D, to match PyTorch's grid_sample expectations.

        if len(shape) == 3:
            d, h, w = shape

            z = torch.linspace(-1, 1, steps=d)
            y = torch.linspace(-1, 1, steps=h)
            x = torch.linspace(-1, 1, steps=w)

            grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')

            identity_grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        else:
            h, w = shape

            y = torch.linspace(-1, 1, steps=h)
            x = torch.linspace(-1, 1, steps=w)

            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

            identity_grid = torch.stack((grid_x, grid_y), dim=-1)

        return identity_grid


    def scale_and_square(self, v: torch.Tensor, N: int | None = None, superres:bool=False) -> torch.Tensor | list[torch.Tensor]:
        """Scale and square the velocity field to obtain the deformation field.

        Parameters:
        - v: Velocity field tensor.
        - N: Number of integration steps.
        - superres: Whether to return a temporally super-resolved list of deformation fields. If True, the output will be a list of tensors representing the deformation at different time steps. Default is False.

        Returns:
        - phi: The diffeomorphic flow obtained by integrating v.
        """


        if N is None:
            N = self.N

        v = self._scale(v, N)  # Scale the velocity field for the scaling and squaring method of integration

        assert self.id is not None, "Identity grid (self.id) must be initialized."

        phi = self.id - v  # Initial deformation is the identity grid minus the velocity field at time step t

        if superres: # Return all intermediate deformation fields
            phi_list = [phi.clone()]

            for _ in range(N-1):
                phi = self._square(phi)
                phi_list.append(phi.clone())

            return phi_list

        for _ in range(N-1):
            phi = self._square(phi)  # Square the deformation N times to obtain the final deformation
        return phi


    def euler(self, v: torch.Tensor, N: int | None = None, superres:bool=False) -> torch.Tensor | list[torch.Tensor]:
        """Euler integration of the velocity field to obtain the deformation field.

        Parameters:
        - v: Velocity field tensor.
        - N: Number of integration steps.
        - superres: Whether to return a temporally super-resolved list of deformation fields. If True, the output will be a list of tensors representing the deformation at different time steps. Default is False.

        Returns:
        - phi: The diffeomorphic flow obtained by integrating v using Euler's method.
        """
        if N is None:
            N = self.N

        assert self.id is not None, "Identity grid (self.id) must be initialized."

        phi0 = self.id - v / N

        if superres: # Return all intermediate deformation fields
            phi_list = [phi0.clone()]
            phi = phi0.clone()
            for _ in range(N-1):
                phi = self._compose(phi, phi0)  # Euler integration step
                phi_list.append(phi.clone())
            return phi_list

        phi = phi0.clone()
        for _ in range(N-1):
            phi = self._compose(phi, phi0)  # Euler integration step
        return phi


    @overload
    def forward(self, v: torch.Tensor, superres: bool = False) -> torch.Tensor:
        ...

    @overload
    def forward(self, v: torch.Tensor, superres: bool = True) -> list[torch.Tensor]:
        ...

    @overload
    def forward(self, v: list[torch.Tensor], superres: bool = False) -> list[torch.Tensor]:
        ...
    
    @overload
    def forward(self, v: list[torch.Tensor], superres: bool = True) -> list[torch.Tensor]:
        ...

    def forward(self, v: torch.Tensor | list[torch.Tensor], superres: bool = False) -> torch.Tensor | list[torch.Tensor]:
        """Integrate the velocity field to obtain the deformation.
        
        input
        -----
        v : torch.Tensor | list[torch.Tensor]
            The (possibly) time-dependent velocity field to be integrated. If it is a list, it should contain tensors of the same shape which represents the velocity field at different time steps. The tensors can be a 4D tensor for 2D deformation (B, H, W, 2) or a 5D tensor for 3D deformation (B, D, H, W, 3).
        superres : bool, optional
            Whether to return a temporally super-resolved list of deformation fields. If True, the output will be a list of tensors representing the deformation at different time steps. Default is False.
        output
        ------
        torch.Tensor | list[torch.Tensor]
            The resulting deformation field after integrating the velocity field. The shape will match the input velocity field, except for the channel dimension which corresponds to the spatial dimensions (2 for 2D, 3 for 3D).
        
        """
        # Here we would implement the actual integration logic, which typically involves solving an ODE to obtain the deformation from the velocity field.
        
        assert isinstance(v, torch.Tensor) or (isinstance(v, list) and all(isinstance(vi, torch.Tensor) for vi in v)), f"Expected input to be a torch.Tensor or a list of torch.Tensors, but got {type(v)}."
        assert v.dim() in (4, 5) if isinstance(v, torch.Tensor) else all(isinstance(vi, torch.Tensor) and vi.dim() in (4, 5) for vi in v), f"Expected velocity field to have 4 dimensions (B, H, W, 2) for 2D or 5 dimensions (B, D, H, W, 3) for 3D, but got {v.dim() if isinstance(v, torch.Tensor) else v[0].dim()} dimensions."

        if isinstance(v, list): # Solve the cases where the input is a list of velocity fields recursively
            if not superres:
                return [self.forward(vi, superres=superres) for vi in v]
            else: # superres is True
                return list(chain.from_iterable([self.forward(vi, superres=superres) for vi in v])) # list of tensors


        # Check and validate the velocity field dimensions and channels

        self._validate_velocity(v)

        if self.id is None:
            self._init_shape(tuple(v.shape[1:-1]))  # Infer the shape from the input velocity field, as advertised in the constructor warning.

        if self.integration == "scaling_and_squaring":
            return self.scale_and_square(v, superres=superres)
        elif self.integration == "euler":
            return self.euler(v, superres=superres)
        else:
            raise ValueError(f"Unsupported integration method: {self.integration}")


class FlowDeformationOperator(torch.nn.Module):
    """This class implements the flow deformation operator, which takes as input a time dependent velocity field and a template, and outputs a deformed sequence of images. 
    This is implemented by first integrating the velocity field to obtain the deformation field, and then applying the group action to deform the image using the obtained deformation field."""

    def __init__(self, N: int| None=None, action: str="geometric", integration: str="scaling_and_squaring", shape: Tuple[int, ...] | None = None, extent: Tuple[float, ...] | None = None, device: torch.device = torch.device("cpu")):
        """
        Initialize the flow deformation operator.

        Parameters:
        - N: Number of integration steps for the velocity field.
        - action: Type of group action to apply ("geometric" by default).
        - shape: Shape of the input image (tuple of ints).
        - extent: Physical extent of the input image (tuple of floats).
        - device: Torch device to use.
        """
        super().__init__() 

        if N is None:
            print(f"Warning: no value for the number of integration steps N was provided for the class {self.__class__.__name__}. Setting N to 10 by default. This may not be optimal for your application...\n")
            N = 10  # Default number of integration steps for the velocity field


        self.velocity_integrator = VelocityIntegrator(N, integration, shape, device)  # Velocity integrator for integrating the velocity field to obtain the deformation field
        self.group_action = GroupAction(action, device)  # Group action for applying the deformation to the image
        self.N = N  # Store the number of integration steps

        self.action = action  # Store the type of group action to apply
        self.integration = integration  # Store the integration method for the velocity field

        self.shape = shape
        self.extent = extent

        self.device = device
        self.to(self.device)

    def _validate_template_and_velocity(self, x: torch.Tensor, v: torch.Tensor | list[torch.Tensor]) -> None:
        """Validate the input image and velocity field for consistency in terms of dimensions, channels, and batch size.

        Args:
        - x: Input image tensor.
        - v: Velocity field tensor or list of tensors.

        Raises:
        - AssertionError: If the input image and velocity field are not consistent.
        """

        assert isinstance(x, torch.Tensor), f"Expected input image to be a torch.Tensor, but got {type(x)}."
        assert isinstance(v, torch.Tensor) or isinstance(v, list), f"Expected velocity field to be a torch.Tensor or a list of torch.Tensors, but got {type(v)}."

        static = isinstance(v, torch.Tensor)
        v_dim = v.dim() if static else v[0].dim()
        shape = v.shape if static else v[0].shape

        # Extract batch size, number of channels, and spatial dimensions from the velocity field shape.

        B = shape[0]  # Batch size
        C = shape[-1]  # Number of channels in the velocity field (2 for 2D, 3 for 3D)
        spatial_dims = shape[1:-1]  # Extract the spatial dimensions from the velocity field shape

        # Ensure that the velocity field has the correct number of dimensions and channels for either 2D or 3D deformation.

        assert v_dim in (4, 5), f"Expected velocity field to have 4 dimensions for 2D or 5 dimensions for 3D, but got {v_dim} dimensions."
        assert C == 2 if v_dim == 4 else C == 3, f"Expected velocity field to have 2 channels for 2D or 3 channels for 3D, but got {C} channels for {v_dim}D velocity field."


        # Ensure concistency between the input image and the velocity field in terms of batch size and spatial dimensions.
    
        assert (x.dim() == v_dim-1 and x.shape[1:] == spatial_dims) and x.shape[0] == B, f"Expected input image to have {v_dim-1} dimensions {(B, *spatial_dims)} for 2D deformation, but got {x.shape}."

    @overload
    def _normalize(self, v: torch.Tensor) -> torch.Tensor: ...

    @overload
    def _normalize(self, v: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def _normalize(self, v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """This function normalizes the deformation phi to ensure that the values are in the range [-1, 1], which is required for use with PyTorch's grid_sample function. The normalization is done by scaling the deformation values based on the dimensions of the input image."""

        # TODO: think more carefully about how this should be implemented

        assert isinstance(v, torch.Tensor) or (isinstance(v, list) and all(isinstance(vi, torch.Tensor) for vi in v)), f"Expected input to be a torch.Tensor or a list of torch.Tensor, but got {type(v)}."
        
        if isinstance(v, list):
            return [self._normalize(vi) for vi in v]

        dim = v.dim()
        shape = v.shape
        assert dim in (4, 5), f"Expected velocity fields to have 4 dimensions (B, H, W, C) for 2D or 5 dimensions (B, D, H, W, C) for 3D, but got {dim} dimensions {shape}."

        if self.extent is None:
            print("Warning: extent is not set. Using default extent based on shape.")
            extent = tuple(float(s) for s in shape[1:-1])
        else:
            extent = self.extent



        C = shape[-1]  # Number of channels in the deformation field (2 for 2D, 3 for 3D)
        assert C == 2 if dim == 4 else C == 3, f"Expected deformation to have 2 channels for 2D or 3 channels for 3D, but got {C} channels."
        assert len(extent) == dim - 2 and len(extent)==C, f"Expected extent to have {dim-2} elements and match the number of channels {C}, but got {len(extent)}."

        v_norm = v.clone()
        for i in range(C): 
            # NOTE: Normalize to the range [-1, 1] based on the spatial dimensions of the input image. We use extent[i]. 
            # Also note that the spatial dimensions are assumed to be in the order (D, H, W) for 3D or (H, W) for 2D which corresponds to (z, y, x) and (y, x) respectively, 
            # whereas torch.nn.functional.grid_sample expects the channel dimension to use the convention (x, y, z) for 3D or (x, y) for 2D. This is why we reverse the order when normalizing. 
        
            v_norm[..., C-1-i] = 2 * v_norm[..., C-1-i] / extent[i] 
        # clip values to [-1, 1]

        v_norm = torch.clamp(v_norm, -1.0, 1.0)

        return v_norm


    @overload
    def forward(self, x: torch.Tensor, v: torch.Tensor, superres: bool = False) -> torch.Tensor: ...

    @overload
    def forward(self, x: torch.Tensor, v: list[torch.Tensor], superres: bool = False) -> list[torch.Tensor]: ...    

    @overload
    def forward(self, x: torch.Tensor, v: torch.Tensor, superres: bool = True) -> list[torch.Tensor]: ...

    @overload
    def forward(self, x: torch.Tensor, v: list[torch.Tensor], superres: bool = True) -> list[torch.Tensor]: ...


    def forward(self, x: torch.Tensor, v: torch.Tensor | list[torch.Tensor], superres: bool = False) ->torch.Tensor | list[torch.Tensor]:
        """Apply the flow deformation to the input image using the given velocity field.
        
        
        input:
            x: torch.Tensor
                The input image to be deformed. For 2D images, the shape should be (B, H, W). For 3D images, the shape should be (B, D, H, W).
            v: torch.Tensor | list[torch.Tensor]
                The velocity field used to compute the deformation. It is either time dependent (list of Tensors) or static (a single Tensor). For 2D images, all tensors should be 4D: (B, H, W, 2). For 3D images, the tensors should be 5D: (B, D, H, W, 3).
            superres: bool, optional
                Whether to return a temporally super-resolved list of deformation fields. If True, the output will be a list of tensors representing the deformation at different time steps. Default is False.
        output: torch.Tensor | list[torch.Tensor]
                The deformed image (sequence). If v is time-dependent (a list of tensors), the output will be a list of deformed images at each time step. If superres is True, then the output will be a temporally super-resolved list of deformed images. Only if v is a single static tensor and superres is False, the output will be a single deformed image tensor rather than a list.
                For 2D images, the shape will be (B, H, W). For 3D images, the shape will be (B, D, H, W). The first element corresponds to the original image, and the subsequent elements correspond to the deformed images at each time step.
        """

        # Validate the input image and velocity field for consistency in terms of dimensions, channels, and batch size.

        self._validate_template_and_velocity(x, v)
        v = self._normalize(v)  # Normalize the velocity field to ensure that the deformation values are in the range [-1, 1]


        if isinstance(v, list): # Deal with the case where the velocity field is time-dependent (a list of tensors)
            if not superres:
                return [self.forward(x, vt, superres=superres) for vt in v]
            else: 
                return list(chain.from_iterable(self.forward(x, vt, superres=superres) for vt in v)) # Here we need to flatten the list of lists of tensors to a list of tensors


        phi: torch.Tensor | list[torch.Tensor] = self.velocity_integrator(v, superres=superres)  # Integrate the velocity field to obtain the deformation field for the current time step
        y: torch.Tensor | list[torch.Tensor] = self.group_action(x, phi)  # Apply the group action to deform the image using the obtained deformation field

        return y
    