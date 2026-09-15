import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from visualization.static_visualization import StaticVisualization
from pathlib import Path


class DynamicVisualization(StaticVisualization):
    """This is a generic class for dynamic visualization of a 4D volume (B, T, D, H, W), where the time dimension is visualized as an animation."""
    
    def __init__(self, meta_data: dict, batch_idx: int = 0, vmin: float =None, vmax: float=None, save_dir: str = None, setup_save_dir: bool = False, verbose: bool = False)->None:
        super().__init__(meta_data, batch_idx, vmin, vmax)
        self.verbose = verbose

        self.save_dir = self._set_up_save_dir(save_dir) if setup_save_dir else Path(save_dir)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Save directory set to {self.save_dir}. Dynamic visualizations will be saved here.")
        else:
            if self.verbose:
                print("No save directory provided. Dynamic visualizations will not be saved.")


    def _set_up_save_dir(self, save_dir: str = None)->Path:


        save_dir = Path(save_dir) if save_dir is not None else None



        patient_id = self.meta_data.get("patient_id", "unknown_patient")[0] if isinstance(self.meta_data.get("patient_id", "unknown_patient"), list) else self.meta_data.get("patient_id", "unknown_patient")

        assert isinstance(patient_id, str), f"Expected patient_id to be a string, but got {type(patient_id)}. Please check the meta_data format."

        patient_id = patient_id.split("_")[0]

        # Study ID will be the date

        study_id = self.meta_data.get("study_date", "unknown_study")[0] if isinstance(self.meta_data.get("study_date", "unknown_study"), list) else self.meta_data.get("study_date", "unknown_study")

        assert isinstance(study_id, str), f"Expected study_id to be a string, but got {type(study_id)}. Please check the meta_data format."
            
        save_dir = save_dir / patient_id / study_id if save_dir is not None else None

        return save_dir


    def visualize_plane_dynamic(self, x: torch.Tensor, plane: str = "axial", slice_idx: int = None, prefix: str = "")->None:
        """This function createes a dynamic (temporal) gif over the speficied plane and slice."""

        x = x.cpu()  # Move the input tensor to CPU for visualization
        self._init_shape(x.shape)  # Initialize the shape and geometry if not already done

        if slice_idx is None:
            slice_idx = self.slice_indices[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]]

        fig, ax = plt.subplots(figsize=(6, 6))

        def update(t):
            s = self._extract_slice(x, plane=plane, slice_idx=slice_idx, time_bin=t)
            im.set_data(s)
            title.set_text(f"{prefix} Plane {plane}, Slice {slice_idx}, Time Bin {t}")
            return [im, title]

        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            raise ImportError("matplotlib.animation is required for dynamic visualization. Please install it with `pip install matplotlib`.")

        im, title = self.visualize_plane(x, ax=ax, plane=plane, slice_idx=slice_idx, time_bin=0, prefix=prefix)
        
        # Create the animation

        anim = FuncAnimation(fig, update, frames=self.time_bins, blit=True)



        # Save the animation as a gif
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / f"{prefix} dynamic_visualization_{plane}_slice{slice_idx}.gif"

            if self.verbose:    
                print(f"Saving dynamic visualization to {save_path}...")
            anim.save(save_path, writer=PillowWriter(fps=2))
        else:
            plt.show()
        plt.close(fig)  # Close the figure to free up memory after saving the animation
    


    def visualize_slices(self, x: torch.Tensor, plane: str = "axial", time_bin: int = 0, prefix: str = "")->None:
        """This function visualizes all slices_indices for the provided plane and time bin."""

        x = x.cpu()  # Move the input tensor to CPU for visualization
        self._init_shape(x.shape)  # Initialize the shape and geometry if not already done


        # Get number of slices in the provided plane
        plane_to_dim = {"axial": self.d, "coronal": self.h, "sagittal": self.w}
        num_slices = plane_to_dim[plane]


        fig, ax = plt.subplots(figsize=(6, 6))

        def update(i):
            s = self._extract_slice(x, plane=plane, slice_idx=i, time_bin=time_bin)
            im.set_data(s)
            title.set_text(f"{prefix} Plane {plane}, Slice {i}, Time Bin {time_bin}")
            return [im, title]
        
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            raise ImportError("matplotlib.animation is required for dynamic visualization. Please install it with `pip install matplotlib`.")
        

        im, title = self.visualize_plane(x, ax=ax, plane=plane, slice_idx=0, time_bin=time_bin, prefix=prefix)

        # Create the animation

        anim = FuncAnimation(fig, update, frames=num_slices, blit=True)

        # Save the animation as a gif
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / f"{prefix} visualization_slices_{plane}_timebin{time_bin}.gif"
        else:
            save_path = f"{prefix} visualization_slices_{plane}_timebin{time_bin}.gif"   
        
        if self.verbose:
            print(f"Saving dynamic visualization of slices to {save_path}...")
        anim.save(save_path, writer=PillowWriter(fps=num_slices//10))  # Adjust fps to make the animation not too fast or too slow
        plt.close(fig)  # Close the figure to free up memory after saving the animation

    def plot_velocity_field(self, v: torch.Tensor, plane: str = "axial", time_bin: int = 0, prefix: str = "")->None:
        """This function visualizes the velocity field for a specific plane, slice, and time bin."""    

        v = v.cpu()  # Move the input tensor to CPU for visualization
        self._init_shape(v.shape)  # Initialize the shape and geometry if not already done
        v_physical, v_magnitude, v_positive_normalized, v_negative_normalized = self.prepare_velocity_field(v)

        # Visualize the positive and negative parts of the velocity field as RGB images

        # Now we can visualize `v_combined_rgb` using the existing visualization functions (e.g., visualize_plane)
        self.visualize_slices(v_positive_normalized, plane=plane, time_bin=time_bin, prefix=f"{prefix}_v_pos")
        self.visualize_slices(v_negative_normalized, plane=plane, time_bin=time_bin, prefix=f"{prefix}_v_neg")

        self.visualize_plane_dynamic(v_positive_normalized, plane=plane, slice_idx=None, prefix=f"{prefix}_v_pos")
        self.visualize_plane_dynamic(v_negative_normalized, plane=plane, slice_idx=None, prefix=f"{prefix}_v_neg")

    def prepare_velocity_field(self, v: torch.Tensor)-> torch.Tensor:
        """
        This function prepares the velocity field for RGB-visualization. This is done by partitioning it into a positive and a negative part, and normalizing each part to [0, 1] such that (1, 1, 1) (e.g. white) corresponds to no motion.

        input:
            v: torch.Tensor of shape (B, T-1, D, H, W, 3) representing the velocity field, where the last dimension corresponds to the velocity components (vx, vy, vz) for each time bin.

        output:
            v_physical: torch.Tensor of shape (B, T-1, D, H, W, 3) representing the velocity field in physical units, where the last dimension corresponds to the velocity components (vx, vy, vz) for each time bin.
            v_magnitude: torch.Tensor of shape (B, T-1, D, H, W) representing the magnitude of the velocity field.
            v_positive: torch.Tensor of shape (B, T-1, D, H, W, 3) representing the positive part of the velocity field, normalized to [0, 1].
            v_negative: torch.Tensor of shape (B, T-1, D, H, W, 3) representing the negative part of the velocity field, normalized to [0, 1].
        """

        assert v.dim() == 6, f"Expected velocity field to have 6 dimensions (B, T-1, D, H, W, 3), but got {v.shape}."
        assert v.shape[-1] == 3, f"Expected last dimension of velocity field to be 3 (corresponding to the velocity components vx, vy, vz), but got {v.shape[-1]}."

        def _convert_to_physical_units(v: torch.Tensor)-> torch.Tensor:
            """This function converts the velocity field from the model's output units to physical units (e.g., mm/s). The specific conversion will depend on the units used in the model and the desired output units. This is a placeholder implementation and should be customized based on the requirements of the project."""
            
            scale_z, scale_xy = self.meta_data.get("resampled_slice_thickness"), self.meta_data.get("resampled_pixel_spacing")

            if isinstance(scale_z, list):
                scale_z = torch.tensor(scale_z).to(v.device).squeeze()  # Convert to tensor and remove any singleton dimensions

            if isinstance(scale_xy, list):
                scale_xy = torch.tensor(scale_xy).to(v.device).squeeze()  # Convert to tensor and remove any singleton dimensions

            scale = torch.cat([scale_z, scale_xy], dim=0)

            scale = scale.view(1, 1, 1, 1, 1, 3)  # Shape: (1, 1, 1, 1, 1, 3)
            
    

            v_physical = v * scale  # Convert to physical units by scaling with the voxel size (assuming the model's output is in voxel units)

            return v_physical
        
        def _magnitude(v: torch.Tensor)-> torch.Tensor:
            """This function computes the magnitude of the velocity field."""
            return torch.sqrt(torch.sum(v**2, dim=-1))  # Shape: (B, T-1, D, H, W)

        v_physical = _convert_to_physical_units(v)
        v_magnitude = _magnitude(v_physical)

        v_positive = torch.clamp(v_physical, min=0.0)
        v_negative = -torch.clamp(v_physical, max=0.0)

        def normalize(v_part):
            max_val = torch.max(v_magnitude)
            if max_val > 0:
                return 1 - v_part / max_val  # Normalize to [0, 1]
            else:
                return 1 - v_part  # If the magnitude is zero, return the original tensor to avoid division by zero

        v_positive_normalized = normalize(v_positive)
        v_negative_normalized = normalize(v_negative)

        return v_physical, v_magnitude, v_positive_normalized, v_negative_normalized
    

    def classify_plane_dynamic(self, x: torch.Tensor, plane: str = "axial", slice_idx: int = None, prefix: str = "") -> str | None:
        """
        Show dynamic visualization and collect quality classification via on-figure buttons.

        Returns:
            "g", "b", "u" if user clicked Good/Bad/Uncertain respectively,
            or None if the window was closed without selection.
        """

        x = x.cpu()  # Move the input tensor to CPU for visualization
        self._init_shape(x.shape)  # Initialize the shape and geometry if not already done

        if slice_idx is None:
            slice_idx = self.slice_indices[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]]

        fig, ax = plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(bottom=0.2)

        im, title = self.visualize_plane(x, ax=ax, plane=plane, slice_idx=slice_idx, time_bin=0, prefix=prefix)

        def update(t):
            s = self._extract_slice(x, plane=plane, slice_idx=slice_idx, time_bin=t)
            im.set_data(s)
            title.set_text(f"{prefix} Plane {plane}, Slice {slice_idx}, Time Bin {t}")
            return [im, title]

        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise ImportError("matplotlib.animation is required for dynamic visualization. Please install it with `pip install matplotlib`.")

        # Use blit=False here to keep widget/button redraw stable across backends.
        anim = FuncAnimation(fig, update, frames=self.time_bins, blit=False)

        # Classification state
        selected = {"value": None}

        # Button axes (left, bottom, width, height)
        ax_good = fig.add_axes([0.16, 0.06, 0.18, 0.08])
        ax_bad = fig.add_axes([0.41, 0.06, 0.18, 0.08])
        ax_uncertain = fig.add_axes([0.66, 0.06, 0.18, 0.08])

        btn_good = Button(ax_good, "High (h)")
        btn_bad = Button(ax_bad, "Low (l)")
        btn_uncertain = Button(ax_uncertain, "Medium (m)")

        def _select(value: str):
            selected["value"] = value
            anim.event_source.stop()
            plt.close(fig)

        btn_good.on_clicked(lambda _evt: _select("h"))
        btn_bad.on_clicked(lambda _evt: _select("l"))
        btn_uncertain.on_clicked(lambda _evt: _select("m"))

        fig.canvas.draw_idle()
        plt.show(block=True)
        return selected["value"]