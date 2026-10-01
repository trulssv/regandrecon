"""Transforms applied by the data pipeline (volumes: normalized range -> attenuation values in mm^-1) and the quality loader (projections: Poisson noise)."""
import torch
from typing import Sequence, Tuple, List, overload

from operators.tomo.ray_trafo import add_poisson_noise, photon_count

class AffineTransform(torch.nn.Module):
    def __init__(self, domain: Tuple[float, float], range: Tuple[float, float]) -> None:
        super().__init__()

        assert len(domain) == 2 and all(isinstance(x, float) for x in domain), "Domain must be a tuple of two floats."
        assert len(range) == 2 and all(isinstance(x, float) for x in range), "Range must be a tuple of two floats."

        super().__init__()
        self.domain_min = domain[0]
        self.domain_max = domain[1]
        self.range_min = range[0]
        self.range_max = range[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the affine transformation to the volume
        # This is a placeholder; actual implementation depends on the specific requirements
        # Apply the affine transformation: scale and shift the volume from the domain to the range
        x = (x - self.domain_min) / (self.domain_max - self.domain_min) # map to [0, 1]
        x = x * (self.range_max - self.range_min) + self.range_min      # map to the target range
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the inverse affine transformation: scale and shift the volume from the range back to the domain
        x = (x - self.range_min) / (self.range_max - self.range_min)
        x = x * (self.domain_max - self.domain_min) + self.domain_min
        return x


class HUTransform(torch.nn.Module):
    """Transforms from attenuation values to Hounsfield Units (HU)."""

    def __init__(self, mu_wa_mm: float | None = None, mu_air_mm: float | None = None) -> None:
        super().__init__()
        self.mu_wa_mm = mu_wa_mm
        self.mu_air_mm = mu_air_mm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mu_wa_mm is None:
            raise ValueError("mu_wa_mm is not specified.")
        if self.mu_air_mm is None:
            # set to zero for air
            self.mu_air_mm = 0.0
        return  1000.0 * (x - self.mu_wa_mm) / (self.mu_wa_mm - self.mu_air_mm)

class InverseHUTransform(torch.nn.Module):
    """Transforms from Hounsfield Units (HU) back to attenuation values."""
    def __init__(self, mu_wa_mm: float | None = None, mu_air_mm: float | None = None) -> None:
        super().__init__()
        self.mu_wa_mm = mu_wa_mm
        self.mu_air_mm = mu_air_mm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mu_wa_mm is None:
            raise ValueError("mu_wa_mm is not specified.")
        if self.mu_air_mm is None:
            # set to zero for air
            self.mu_air_mm = 0.0

        return (self.mu_wa_mm - self.mu_air_mm) * (x / 1000.0) + self.mu_wa_mm
class ResizeTransform(torch.nn.Module):
    """Resizes the input volume to the specified size.
    input:
        x (torch.Tensor): The input volume to be resized. Must have 3 (B, H, W) dimensions for a 2D size or 4 (B, D, H, W) dimensions for a 3D size.
    output:
        torch.Tensor: The resized volume with the same number of dimensions as the input.

    """

    MODES = {2: "bilinear", 3: "trilinear"}

    def __init__(self, size: Sequence[int]) -> None:
        super().__init__()
        assert len(size) in self.MODES, f"Size must have 2 (H, W) or 3 (D, H, W) entries but got {size}."
        self.size = tuple(size)
        self.mode = self.MODES[len(size)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the ResizeTransform module.

        Args:
            x (torch.Tensor): The input volume to be resized.

        Returns:
            torch.Tensor: The resized volume. This is achieved using bilinear (2D) or trilinear (3D) interpolation with torch.nn.functional.interpolate,
            where the batch dimension takes the place of the channel dimension.
        """
        assert x.dim() == len(self.size) +1, f"Input tensor dimensions must have length equal to the size plus one but got {x.dim()} and expected {len(self.size) + 1}."
        return torch.nn.functional.interpolate(x.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False).squeeze(0)



class DataTransform(torch.nn.Module):
    """This class performs the preprocessing pipeline for volumetric data. This consists of two steps: 
    1. resizing the volume to the specified size.
    2. Applying an affine transformation to map the data from the normalized range to attenuation values in mm^-1."""
    def __init__(self, normalized_range: Sequence[float], hu_range: Sequence[float], mu_water_mm: float, mu_air_mm: float, size: Sequence[int], device: torch.device = torch.device("cpu")):
        super().__init__()
        assert len(normalized_range) == 2, "normalized_range must have two entries."
        assert len(hu_range) == 2, "hu_range must have two entries."
        assert len(size) in (2, 3), "size must have two (H, W) or three (D, H, W) entries."

        self.normalized_range = (float(normalized_range[0]), float(normalized_range[1]))
        self.hu_range = (float(hu_range[0]), float(hu_range[1]))
        self.mu_water_mm = float(mu_water_mm)
        self.mu_air_mm = float(mu_air_mm)
        self.shape = tuple(int(s) for s in size)
        self.device = device


        # Maps from the normalized range to the HU range using an affine transformation.

        self.affine_transform = AffineTransform(
            domain=self.normalized_range,
            range=self.hu_range
        )

        # Converts from HU to attenuation values in mm^-1.

        self.inv_hu_transform = InverseHUTransform(mu_wa_mm=self.mu_water_mm, mu_air_mm=self.mu_air_mm) # Convert from HU to attenuation values in mm^-1.

        # Resizes the volume to the specified size.

        self.resize = ResizeTransform(size=self.shape)

    @classmethod
    def from_config(cls, preprocess_cfg: dict, device: torch.device = torch.device("cpu")) -> "DataTransform":
        """Builds the DataTransform from data/configs/preprocess.json."""
        return cls(
            normalized_range=preprocess_cfg["INITIAL_RANGE"],
            hu_range=preprocess_cfg["HU_RANGE"],
            mu_water_mm=preprocess_cfg["mu_wa_mm"],
            mu_air_mm=preprocess_cfg["mu_air_mm"],
            size=preprocess_cfg["SHAPE"],
            device=device,
        )

    def validate_input(self, x: torch.Tensor, size: Tuple[int, int, int]) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor instance.")

        assert x.dim() in (3, 4), "Input tensor must have 3 or 4 dimensions (B, H, W) or (B, D, H, W)."

        if x.dim() == 3:
            assert len(size) == 2, "Size must be a tuple of two integers for 3D input."

        if x.dim() == 4:
            B, D, H, W = x.shape

        if x.dim() == 4:
            assert len(size) == 3, "Size must be a tuple of three integers for 4D input."

    @overload
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]: ...

    def forward(self, x: torch.Tensor | List[torch.Tensor]) -> torch.Tensor | List[torch.Tensor]:

        if isinstance(x, list):
            assert all(isinstance(xi, torch.Tensor) for xi in x), "All elements of the input list must be torch.Tensor instances."
            return [self.forward(xi) for xi in x]

        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor instance."

        x = x.clone().detach() # Ensure the input tensor is not modified in-place.
        x = x.to(torch.float32) # Ensure the input tensor is in float32 format.
        # Ensure the input tensor is contiguous in memory.
        x = x.contiguous()
        # Ensure the input tensor is on the correct device.
        x = x.to(self.device)



        x = self.resize(x) # Resize the volume to the specified size.
        x = self.affine_transform(x) # Apply an affine transformation to map the data from the normalized range to the HU range.
        x = self.inv_hu_transform(x) # Convert from HU to attenuation values in mm^-1.
        return x


class PoissonNoise:
    """Adds CT Poisson noise to clean projections (line integrals of attenuation values in mm^-1), see operators.tomo.ray_trafo.add_poisson_noise.
    A new noise realization is drawn on every call, i.e. every epoch."""
    def __init__(self, flux: float, nViews: int, nDetectorCols: int, nDetectorRows: int | None = None, epsilon: float = 1e-6) -> None:
        self.n0 = photon_count(flux, nViews, nDetectorCols, nDetectorRows)
        self.epsilon = epsilon

    @classmethod
    def from_config(cls, ray_trafo_cfg: dict, flux: float | None = None) -> "PoissonNoise":
        """Builds the noise model from data/configs/ray_trafo/<geometry>.json. The flux (dose) defaults to the one in the config."""
        return cls(
            flux=ray_trafo_cfg["Flux"] if flux is None else flux,
            nViews=ray_trafo_cfg["nViews"],
            nDetectorCols=ray_trafo_cfg["nDetectorCols"],
            nDetectorRows=ray_trafo_cfg.get("nDetectorRows"),
        )

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return add_poisson_noise(y, n0=self.n0, epsilon=self.epsilon)
