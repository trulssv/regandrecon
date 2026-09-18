import torch

from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.axes import Axes

class StaticVisualization:
    """This is a generic class for static visualization of a 4D volume (B, T, D, H, W), where """

    def __init__(self, meta_data: dict, batch_idx: int = 0, vmin: float|None =None, vmax: float|None =None)->None:

        
        self.meta_data = meta_data
        self.batch_idx = batch_idx
        self.vmin = vmin
        self.vmax = vmax

            
    def _init_shape(self, sz: torch.Size)->None:
        """This function initializes the shape of the input tensor and checks that it has the expected dimensions. It also calculates the slice indices for the middle slices in each plane (axial, coronal, sagittal) for visualization purposes."""
        self.sz = sz

        assert len(self.sz) in (5, 6), f"Expected input tensor to have 5 or 6 dimensions (B, T, D, H, W), but got {len(self.sz)} dimensions."

        if len(self.sz) == 6:
            b, t, d, h, w, c = self.sz
            assert c == 3, f"Expected input tensor to have 3 channels for RGB, but got {c} channels."

        else:
            b, t, d, h, w = self.sz
            c = None


        assert self.batch_idx < b, f"Batch index {self.batch_idx} is out of bounds for batch size {b}."


        self.slice_indices = [d // 2, h // 2, w // 2] 


        self.d , self.h , self.w = d, h, w
        self.time_bins = t
        self.has_channels = c is not None

        self._get_geometry()



    def _get_geometry(self):

        assert "resampled_pixel_spacing" in self.meta_data, "Expected 'resampled_pixel_spacing' in meta_data for correct aspect ratio visualization, but it is not present. Please check the meta_data format."
        assert "resampled_slice_thickness" in self.meta_data, "Expected 'resampled_slice_thickness' in meta_data for correct aspect ratio visualization, but it is not present. Please check the meta_data format."

        pixel_spacing = self.meta_data["resampled_pixel_spacing"]
        slice_thickness = self.meta_data["resampled_slice_thickness"]

        self.X: float = self.w * pixel_spacing[0]
        self.Y: float = self.h * pixel_spacing[1]
        self.Z: float = self.d * slice_thickness


    def _extract_volume(self, x: torch.Tensor, time_bin: int)->torch.Tensor:
        volume = x[self.batch_idx, time_bin]
        return volume


    def _extract_slice(self, x: torch.Tensor, plane:str="axial", slice_idx:int|None =None, time_bin:int=0)->torch.Tensor:
        
        def _extent(plane: str)->list[float]:
            if plane == "axial":
                return [0, self.Y, 0, self.X]
            elif plane == "coronal":
                return [0, self.Z, 0, self.X]
            elif plane == "sagittal":
                return [0, self.Z, 0, self.Y]
            else:
                raise ValueError(f"Invalid plane {plane}. Expected one of ['axial', 'coronal', 'sagittal'].")

        vol = self._extract_volume(x, time_bin)
        
        plane_to_dim = {"axial": 0, "coronal": 1, "sagittal": 2}

        assert plane in plane_to_dim, f"Invalid plane {plane}. Expected one of {list(plane_to_dim.keys())}."

        if slice_idx is None:
            slice_idx = self.slice_indices[plane_to_dim[plane]]

        if plane == "axial":
            s = vol[slice_idx, :, :,...]
        elif plane == "coronal":
            s = vol[:, slice_idx, :, ...]
        elif plane == "sagittal":
            s = vol[ :, :, slice_idx, ...]
        extent = _extent(plane)

        # TODO Implement extent into the visualization pipeline to ensure correct aspect ratio and physical dimensions. This will require passing the extent to the plotting functions and using it in the imshow calls.

        return s    
  


    def visualize_plane(self, x:torch.Tensor, ax: Axes, plane:str="axial", slice_idx:int|None=None, time_bin: int=0, prefix:str = "", save: bool = False)->tuple[AxesImage, Text]:
        


        s = self._extract_slice(x, plane=plane, slice_idx=slice_idx, time_bin=time_bin)


        if self.has_channels:
            assert s.dim() == 3, f"Expected slice to have 3 dimensions (C, H, W), but got {s.dim()} dimensions."
            im, title = self.plot_plane_rgb(s, ax, plane=plane, slice_idx=slice_idx, time_bin=time_bin, prefix=prefix)
        else: 
            assert s.dim() == 2, f"Expected slice to have 2 dimensions (H, W), but got {s.dim()} dimensions."
            im, title = self.plot_plane_grayscale(s, ax, plane=plane, slice_idx=slice_idx, time_bin=time_bin, prefix=prefix)

        if save:
            plt.savefig(f"{prefix}_plane_{plane}_slice_{slice_idx}_time_bin_{time_bin}.png", bbox_inches='tight')
            print(f"Saved visualization to {prefix}_plane_{plane}_slice_{slice_idx}_time_bin_{time_bin}.png")

        return im, title
        


    def plot_plane_rgb(self, x: torch.Tensor, ax: Axes, plane="axial", slice_idx=None, time_bin=None, prefix:str = "")->tuple[AxesImage, Text]:
        """This function plots a 2D slice with 3 channels (C, H, W) as an RGB image."""


        if ax is None:
            print("No axis provided for plotting. Creating a new figure and axis.")
            fig, ax = plt.subplots(figsize=(6, 6))

        assert x.dim() == 3, f"Expected input tensor to have 3 dimensions (C, H, W), but got {x.dim()} dimensions."

        if slice_idx is None:
            slice_idx = self.slice_indices[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]]

        im = ax.imshow(x)
        ax.axis("off")

        title = ax.set_title(f"{prefix} Plane {plane}, Slice {slice_idx}, Time Bin {time_bin}")
        return im, title

    def plot_plane_grayscale(self, x: torch.Tensor, ax: Axes, plane="axial", slice_idx=None, time_bin=None, prefix:str = "")->tuple[AxesImage, Text]:
        """This function plots a 2D slice with 1 channel (H, W) as a grayscale image."""

        if ax is None:
            print("No axis provided for plotting. Creating a new figure and axis.")
            fig, ax = plt.subplots(figsize=(6, 6))

        assert x.dim() == 2, f"Expected input tensor to have 2 dimensions (H, W), but got {x.dim()} dimensions."

        if slice_idx is None:
            slice_idx = self.slice_indices[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]]

        # Physical dimensions

        # TODO fix a helper function that returns the physical dimension for the given plane. 

        extent = [self.Z, self.Y, self.X]
        del extent[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]] 
        extent = [0, float(extent[0]), 0, float(extent[1])]
        
        assert len(extent) == 4, f"Expected extent to have 4 elements for 2D visualization, but got {len(extent)} elements." 


        # Plot the slice with correct aspect ratio and physical dimensions

        im = ax.imshow(x, cmap="gray", vmin=self.vmin, vmax=self.vmax, aspect='equal', origin='lower', extent=extent) # type ignore
        # plt.colorbar()  # Add a colorbar to show the intensity scale
        ax.axis("off")
        title = ax.set_title(f"{prefix} Plane {plane}, Slice {slice_idx}, Time Bin {time_bin}")
        return im, title


