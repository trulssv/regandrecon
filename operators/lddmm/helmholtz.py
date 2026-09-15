import torch
from torch.fft import rfftn, irfftn,rfftfreq, fftfreq
import torch.nn as nn
from typing import overload


class HelmholtzOperator(nn.Module):
    r"""This class implements the Helmholtz operator, which is used in the HLPD model to regularize the velocity field. 
    The Helmholtz operator is defined as $L = (\\gamma I - \\alpha^2 \Delta)^\\beta$, where $I$ is the identity operator, $\\alpha$ is a regularization parameter, 
    and $\\Delta$ is the Laplacian operator. The inverse of the Helmholtz operator can be efficiently computed in the Fourier domain, 
    which allows for fast regularization of the velocity field during training. Extent specifies the physical size of the spatial dimensions.
    
    input:
        v (torch.Tensor or list[torch.Tensor]): The input velocity field(s) to which the Helmholtz operator will be applied. 
            For a single velocity field, the shape should be (B, H, W, C) for 2D or (B, D, H, W, C) for 3D, 
            where B is the batch size, D is the depth (for 3D), H is the height, W is the width, and C is the number of velocity components (2 for 2D, 3 for 3D).
        return_fft (bool, optional): If True, the forward method will return the Fourier transform of the regularized velocity field instead of the spatial domain representation. Default is False.
    output (torch.Tensor): The regularized velocity field. If return_fft is True, this will be the Fourier transform of the regularized velocity field; otherwise, it will be the spatial domain representation.
    
    """
    def __init__(self, dim: int, alpha: float = 1.0, gamma: float = 1.0, beta: float = 1.0, extent: tuple[tuple[float, float], ...] | None= None, return_fft: bool = False, device: torch.device = torch.device("cpu"))->None:
        super().__init__()

        assert dim in (2, 3), f"Expected dim to be 2 or 3, but got {dim}."

        self.dim = dim
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.extent = extent
        self.return_fft = return_fft
        self.device = device
        self.to(self.device)
    
        if extent is not None:
            assert len(extent) == self.dim, f"Expected extent to have length {self.dim}, but got {len(extent)}."
            assert all(isinstance(e, tuple) and len(e) == 2 and all(isinstance(x, float) for x in e) for e in extent), "Each extent must be a tuple of two floats."

    def compute_frequency_grid(self, shape: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
        """Compute the frequency grid for the given spatial shape and device.

        Args:
            device (torch.device): The device on which to create the frequency grid.

        Returns:
            torch.Tensor: The frequency grid for the given spatial shape and device.
        """

        assert len(shape) == self.dim, f"Expected shape to have length {self.dim}, but got {len(shape)}."
        
        # Get sampling units from the extent if provided

        extent = self.extent if self.extent else tuple((0.0, 1.0) for _ in range(len(shape)))

        assert len(extent) == self.dim, f"Expected extent to have length {self.dim}, but got {len(extent)}."
        assert all(isinstance(e, tuple) and len(e) == 2 and all(isinstance(x, float) for x in e) for e in extent), "Each extent must be a tuple of two floats."
        spacing = tuple((e[1] - e[0])/ s for e, s in zip(extent, shape))


        if self.dim == 2:
            H, W = shape
            kx = rfftfreq(W, d=spacing[1]).to(device)
            ky = fftfreq(H, d=spacing[0]).to(device)
            KX, KY = torch.meshgrid(ky, kx, indexing='ij')
            return KX, KY
        else:  # self.dim == 3
            D, H, W = shape
            kx = rfftfreq(W, d=spacing[2]).to(device)
            ky = fftfreq(H, d=spacing[1]).to(device)
            kz = fftfreq(D, d=spacing[0]).to(device)
            KX, KY, KZ = torch.meshgrid(kz, ky, kx , indexing='ij')
            return KX, KY, KZ

    @overload
    def forward(self, v: torch.Tensor) -> torch.Tensor: ...

    @overload
    def forward(self, v: list[torch.Tensor]) -> list[torch.Tensor]: ...


    def forward(self, v: torch.Tensor|list[torch.Tensor])->torch.Tensor|list[torch.Tensor]:
        """Apply the Helmholtz operator to the input velocity field.

        If the input is a list of velocity fields, the operator is applied to each element in the list.

        input:
            v (torch.Tensor or list[torch.Tensor]): The input velocity field(s) to which the Helmholtz operator will be applied. 
                For a single velocity field, the shape should be (B, H, W, C) for 2D or (B, D, H, W, C) for 3D, 
                where B is the batch size, D is the depth (for 3D), H is the height, W is the width, and C is the number of velocity components (2 for 2D, 3 for 3D).
             
        output:
            torch.Tensor or list[torch.Tensor]: The regularized velocity field(s) after applying the Helmholtz operator. 
                The shape will be the same as the input velocity field(s).

        """

        assert isinstance(v, torch.Tensor) or (isinstance(v, list) and all(isinstance(vi, torch.Tensor) for vi in v)), f"Expected input to be a torch.Tensor or a list of torch.Tensor, but got {type(v)}."

        if isinstance(v, list):
            return [self.forward(vi) for vi in v]

        dim = v.dim()
        shape = v.shape # (B, H, W, C) for 2D or (B, D, H, W, C) for 3D
        C = shape[-1]
        B = shape[0]
        assert dim ==self.dim + 2 and C ==self.dim, f"Expected input velocity field to have {self.dim + 2} dimensions and the last dimension to have size {self.dim}, but got {dim} dimensions with shape {shape}."

        v_fft = rfftn(v, dim=tuple(range(1, 1 + self.dim))) # Compute the Fourier transform along the spatial dimensions for 2D and 3D velocity fields
        
        # Create a grid of frequencies for each spatial dimension
        spatial_shape = v.shape[1:-1]
    
        Kgrid = self.compute_frequency_grid(shape=spatial_shape, device=v.device) # Computes a 2D or 3D frequency grid depending on the spatial dimensions with with adjusted spacing based on the extent.
        KSquaredgrid = (torch.sum(torch.stack([K**2 for K in Kgrid], dim=0), dim=0)).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # Shape: (1, 1, D, H, W, 1) for 3D or (1, 1, H, W, 1) for 2D

 
        # Compute the squared magnitude of the frequency grid
      

        # Apply the Helmholtz operator in the Fourier domain
        L_fft = (self.alpha + self.gamma * (4 * torch.pi**2) * KSquaredgrid).pow(self.beta)
        
        # Regularize the velocity field by multiplying with the inverse Helmholtz operator in the Fourier domain

        v_reg_fft = v_fft * L_fft

        if self.return_fft:
            return v_reg_fft 

        # Compute the inverse Fourier transform to get the regularized velocity field in the spatial domain
        v_reg = irfftn(v_reg_fft, dim=tuple(range(1, 1 + self.dim)))
        
        return v_reg