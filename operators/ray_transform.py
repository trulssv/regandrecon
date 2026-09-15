import odl
import odl.applications.tomo as odl_tomo
import numpy as np
from typing import Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from odl.contrib.torch import OperatorModule


class RayTransform:
    def __init__(self, ray_trafo_cfg: dict, preprocess_cfg: dict, meta_data: dict = None, init_transforms:bool=True) -> None:
        
        # Physics & Geometry parameters

        self.ray_trafo_cfg = ray_trafo_cfg          # Parameters relating to the ray transform
        self.preprocess_cfg = preprocess_cfg        # Parameters relating to the preprocessing protocol
        self.meta_data = meta_data                  # Meta data containing information about the physical dimensions of the volume for correct aspect ratio visualization



        self.shape = ray_trafo_cfg.get("shape")
        self.axis = ray_trafo_cfg.get("axis")
        self.num_angles = ray_trafo_cfg.get("num_angles")
        self.geometry = ray_trafo_cfg.get("geometry", "parallel")

        # Initialize the ray transform, adjoint transform, and FBP transform based on the specified geometry and parameters in ray_trafo_cfg. We only initialize the ray transform and the corresponding adjoint and FBP transforms if init_transforms is True, otherwise we only set up the geometry parameters and leave the initialization of the transforms to be done later. This is useful for the DynamicRayTransform class, where we need to construct a separate ray transform for each time bin with a different geometry, so we want to delay the initialization of the transforms until we have constructed all the geometries for each time bin.

        if init_transforms:
            self.ray_trafo = self.make_ray_transform(shape=self.shape)
            self.adjoint_trafo = self.make_adjoint_transform()
            self.fbp_trafo = self.make_fbp_transform()

            # Convert to torch OperatorModule

            self.ray_trafo_module = OperatorModule(self.ray_trafo)
            self.adjoint_trafo_module = OperatorModule(self.adjoint_trafo)
            self.fbp_trafo_module = OperatorModule(self.fbp_trafo)
    
    def _get_physical_dimensions(self) -> Tuple[Tuple[float, float, float], Tuple[int, int, int]]:

        # Get the physical dimensions of the volume

        if self.meta_data is not None:

            assert "resampled_pixel_spacing" in self.meta_data, "Expected 'resampled_pixel_spacing' in meta_data for correct aspect ratio visualization, but it is not present. Please check the meta_data format."
            assert "resampled_slice_thickness" in self.meta_data, "Expected 'resampled_slice_thickness' in meta_data for correct aspect ratio visualization, but it is not present. Please check the meta_data format."

            pixel_spacing = self.meta_data["resampled_pixel_spacing"]
            slice_thickness = self.meta_data["resampled_slice_thickness"]

        else: # Load global voxel parameters

            pixel_spacing = self.preprocess_cfg.get("pixel_spacing_mm")
            slice_thickness = self.preprocess_cfg.get("slice_thickness_mm")

        s_x, s_y, s_z = pixel_spacing[0].item() if isinstance(pixel_spacing[0], torch.Tensor) else pixel_spacing[0], pixel_spacing[1].item() if isinstance(pixel_spacing[1], torch.Tensor) else pixel_spacing[1], slice_thickness.item() if isinstance(slice_thickness, torch.Tensor) else slice_thickness


        # Get shape

        assert "SHAPE" in self.preprocess_cfg, "Expected 'SHAPE' in preprocess_cfg for correct aspect ratio visualization, but it is not present. Please check the preprocess_cfg format."

        shape = self.preprocess_cfg["SHAPE"] # D, H, W

        shape = list(shape) # Convert to list to allow for modification if needed (e.g., reordering dimensions, etc.)
        # Calculate physical dimensions in mm

        Z, Y, X = shape[0] * s_z, shape[1] * s_y, shape[2] * s_x

        volume_extent_mm = (Z, Y, X)

        # Load detector extent

        if "detector_extent_mm" in self.ray_trafo_cfg.keys():

            detector_extent_mm = self.ray_trafo_cfg["detector_extent_mm"]   
        else:
            print("Warning: 'detector_extent_mm' not found in ray_trafo_cfg. Using volume extent for detector extent, which may lead to incorrect geometry construction if the detector extent is different from the volume extent. Please check the ray_trafo_cfg format and provide 'detector_extent_mm' if the detector extent is different from the volume extent.")
            detector_extent_mm = volume_extent_mm[:2]

        return volume_extent_mm, detector_extent_mm, shape

    def _get_reconstruction_space(self, extent: Tuple[float, float, float], shape: Tuple[int, int, int]) -> odl.uniform_discr:


        # Get the reconstruction space for the ray transform

        min_pt = [-extent[0] / 2, -extent[1] / 2, -extent[2] / 2]  # z, y, x
        max_pt = [extent[0] / 2, extent[1] / 2, extent[2] / 2]     # z, y, x

        reco_space = odl.uniform_discr(
            min_pt=min_pt,
            max_pt=max_pt,
            shape=shape,
            dtype="float32",
        )

        return reco_space

    def _get_detector_geometry(self, extent: Tuple[float, float, float], angle_partition: odl.uniform_partition = None) -> odl_tomo.Geometry:
        
        # Get the detector partition for the ray transform

        assert "detector_shape" in self.ray_trafo_cfg, "Expected 'detector_shape' in ray_trafo_cfg for correct geometry construction, but it is not present. Please check the ray_trafo_cfg format."

        detector_shape = self.ray_trafo_cfg["detector_shape"] # in mm


        detector_partition = odl.uniform_partition([-extent[0] / 2, -extent[1] / 2], [extent[0] / 2, extent[1] / 2], detector_shape) 

        if angle_partition is None:

            # Angle partition was not provided, construct it based on the number of angles specified in ray_trafo_cfg

            assert "num_angles" in self.ray_trafo_cfg, "Expected 'num_angles' in ray_trafo_cfg for correct geometry construction, but it is not present. Please check the ray_trafo_cfg format."

            num_angles = self.ray_trafo_cfg["num_angles"]
            angle_partition = odl.uniform_partition(0.0, float(np.pi), int(num_angles))

        # Get axis of rotation

        assert "axis" in self.ray_trafo_cfg, "Expected 'axis' in ray_trafo_cfg for correct geometry construction, but it is not present. Please check the ray_trafo_cfg format."

        axis = self.ray_trafo_cfg.get("axis") # 0 for axial, 1 for coronal, 2 for sagittal since we have the volume in z, y, x order in ODL

        # Construct detector geometry based on the specified geometry type in ray_trafo_cfg

        assert "geometry" in self.ray_trafo_cfg, "Expected 'geometry' in ray_trafo_cfg for correct geometry construction, but it is not present. Please check the ray_trafo_cfg format."

        geometry = self.ray_trafo_cfg["geometry"]
        assert geometry in self.ray_trafo_cfg.keys(), f"Expected geometry to be one of {self.ray_trafo_cfg.keys()}, but got {geometry}. Please check the ray_trafo_cfg format."

        if geometry == "parallel":
            geometry = odl_tomo.Parallel3dAxisGeometry(
                angle_partition,
                detector_partition,
                axis=axis,
            )
        elif geometry == "cone":

            src_radius = self.ray_trafo_cfg["cone"].get("src_radius")
            det_radius = self.ray_trafo_cfg["cone"].get("det_radius")

            geometry = odl_tomo.ConeBeamGeometry(
                angle_partition,
                detector_partition,
                src_radius=src_radius,
                det_radius=det_radius,
                axis=axis,
            )

        elif geometry == "helical":
            src_radius = self.ray_trafo_cfg["helical"].get("src_radius")
            det_radius = self.ray_trafo_cfg["helical"].get("det_radius")
            pitch = self.ray_trafo_cfg["helical"].get("pitch")

            # We use a different angular partition for the helical geometry to ensure thatwe get multiple rotations. 

            num_angles_per_rotation = self.ray_trafo_cfg["helical"].get("num_angles_per_rotation")
            num_rotations = self.ray_trafo_cfg["helical"].get("num_rotations")
            num_angles = num_angles_per_rotation * num_rotations
            angle_partition = odl.uniform_partition(0.0, float(num_rotations * np.pi * 2), int(num_angles))

            geometry = odl_tomo.ConeBeamGeometry(
                angle_partition,
                detector_partition,
                src_radius=src_radius,
                det_radius=det_radius,
                pitch=pitch,
                axis=axis,
            )

        return geometry

    def make_ray_transform(self, shape: Tuple[int, int, int]) -> odl_tomo.RayTransform:
        
        volume_extent, detector_extent, shape = self._get_physical_dimensions()
        reco_space = self._get_reconstruction_space(extent=volume_extent, shape=shape)
        geometry = self._get_detector_geometry(extent=detector_extent)
        
        return odl_tomo.RayTransform(reco_space, geometry)
    
    def make_adjoint_transform(self):
        return self.ray_trafo.adjoint
    

    def make_fbp_transform(self):
        return odl_tomo.fbp_op(self.ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)
    

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        
        return self.ray_trafo_module(vol)
    
    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        return self.adjoint_trafo_module(sino)
    
    def fbp_reconstruct(self, sino: torch.Tensor) -> torch.Tensor:
        return self.fbp_trafo_module(sino)

    
    def simulate_noise(self, sino: torch.Tensor) -> torch.Tensor:

        # First, we need to resscale the sinogram from normalized intensity in mm to 

        mu_wa_mm = self.ray_trafo_cfg.get("mu_wa_mm")
        N0 = self.ray_trafo_cfg.get("N0")

        lambda_ = N0 * torch.exp(-sino * mu_wa_mm) # Convert from normalized intensity to relative attenuation values

        epsilon = 1e-6

        noise = torch.poisson(lambda_ + epsilon) # Simulate Poisson noise based on the expected number of photons, which is given by lambda_. The Poisson distribution is appropriate for modeling the noise in CT imaging, as it captures the statistical nature of photon counting.

        # We add a small epsilon to the noise to avoid taking the log of zero, which would result in negative infinity. The epsilon value is set to a small number (e.g., 1e-6) to ensure numerical stability while not significantly affecting the noise simulation.
        
        noise = noise 

        sino_noise = -torch.log(noise / N0) / mu_wa_mm # Convert back from relative attenuation values to normalized intensity in mm
        
        return sino_noise

class DynamicRayTransform(RayTransform):
    """
    This class represents a dynamic ray transform, where the time dimension is explicitly modeled in the geometry and the forward operator. The time dimension is discretized into time bins, and the geometry is constructed such that we have a separate set of projection angles for each time bin. 
    This allows us to model dynamic imaging scenarios where the object changes over time and we want to reconstruct a 4D volume (B, T, D, H, W) from the corresponding 4D sinogram (B, T, num_angles_per_time_bin, detector_shape).
    """
    
    
    def __init__(self, ray_trafo_cfg: dict, preprocess_cfg: dict, meta_data: dict = None, init_transforms: bool = True) -> None:
        super().__init__(ray_trafo_cfg, preprocess_cfg, meta_data, init_transforms=False)

        assert "time_bins" in preprocess_cfg, "Expected 'time_bins' in preprocess_cfg for correct geometry construction, but it is not present. Please check the preprocess_cfg format."

        self.time_bins = preprocess_cfg.get("time_bins")

        # We construct a separate ray transform for each time bin, with the corresponding geometry that has a different set of projection angles for each time bin.

        # Initialize the ray transforms, adjoint transforms, and FBP transforms for each time bin

        if init_transforms:

            self.ray_trafo, self.angles = self.make_ray_transform(shape=self.shape)
            self.adjoint_trafo = self.make_adjoint_transform()
            self.fbp_trafo = self.make_fbp_transform()

            # Convert to torch OperatorModule

            self.ray_trafo_module = [OperatorModule(ray_trafo) for ray_trafo in self.ray_trafo]
            self.adjoint_trafo_module = [OperatorModule(adjoint_trafo) for adjoint_trafo in self.adjoint_trafo]
            self.fbp_trafo_module = [OperatorModule(fbp_trafo) for fbp_trafo in self.fbp_trafo]





    def _get_angle_partition(self) -> odl.uniform_partition:

        angle_partition = []  
        start_angles = []

        
        num_angles_per_time_bin = self.ray_trafo_cfg["dynamic"].get("num_angles_per_time_bin") # Angular range for each time bin in degrees
        num_projections_per_time_bin = self.ray_trafo_cfg["dynamic"].get("num_projections_per_time_bin") # Number of projections for each time bin

        for t in range(self.time_bins):
            

            start_angle = t * num_angles_per_time_bin * np.pi / 180  # Convert from degrees to radians
            end_angle = (t + 1) * num_angles_per_time_bin * np.pi / 180  # Convert from degrees to radians

            angles = odl.uniform_partition(start_angle, end_angle, num_projections_per_time_bin)
            angle_partition.append(angles)
            start_angles.append(start_angle)


        return angle_partition, start_angles


    def _initialize_sino(self, batch_size: int, device: torch.device) -> torch.Tensor:
        num_projections_per_time_bin = self.ray_trafo_cfg["dynamic"].get("num_projections_per_time_bin")
        y = torch.empty((batch_size, self.time_bins, num_projections_per_time_bin, self.ray_trafo[0].range.shape[1], self.ray_trafo[0].range.shape[2]), dtype=torch.float32, device=device)
        return y
    
    def _initialize_recon(self, batch_size: int, device: torch.device) -> torch.Tensor:
        recon = torch.empty((batch_size, self.time_bins, self.ray_trafo[0].domain.shape[0], self.ray_trafo[0].domain.shape[1], self.ray_trafo[0].domain.shape[2]), dtype=torch.float32, device=device)
        return recon
        

    def make_ray_transform(self, shape: Tuple[int, int, int]) -> Sequence[odl_tomo.RayTransform]:

        volume_extent, detector_extent, shape = self._get_physical_dimensions()
        reco_space = self._get_reconstruction_space(extent=volume_extent, shape=shape)
        angle_partition, start_angles = self._get_angle_partition()


        ray_transforms = []

        for t in range(self.time_bins):

            geometry = self._get_detector_geometry(extent=detector_extent, angle_partition=angle_partition[t]) 
            ray_transform = odl_tomo.RayTransform(reco_space, geometry)
            ray_transforms.append(ray_transform)

        return ray_transforms, start_angles


    def make_adjoint_transform(self):
        adjoint_transforms = [ray_trafo.adjoint for ray_trafo in self.ray_trafo]
        return adjoint_transforms
    
    def make_fbp_transform(self):
        fbp_transforms = [odl_tomo.fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0) for ray_trafo in self.ray_trafo]
        return fbp_transforms
    
    def __call__(self, vol: torch.Tensor) -> torch.Tensor:

        # We apply the corresponding ray transform for each time bin to the corresponding volume at that time bin. The input volume is expected to have shape (B, T, D, H, W), and the output sinogram will have shape (B, T, num_projections_per_time_bin, detector_shape).

        assert vol.ndim == 5, f"Expected input volume to have 5 dimensions (B, T, D, H, W), but got {vol.ndim} dimensions."
        assert vol.shape[1] == self.time_bins, f"Expected time dimension of input volume to match the number of time bins {self.time_bins}, but got {vol.shape[1]} time bins."

        # Allocate memory for the output sinogram. The shape of the output sinogram will be (B, T, num_projections_per_time_bin, detector_shape), where num_projections_per_time_bin and detector_shape are determined by the geometry of the ray transform for each time bin.

        y = self._initialize_sino(batch_size=vol.shape[0], device=vol.device) # Initialize the sinogram tensor with the appropriate shape and data type. The shape of the sinogram will be (B, T, num_angles_per_time_bin, detector_shape), where num_angles_per_time_bin and detector_shape are determined by the geometry of the ray transform for each time bin. The data type is set to float32 to ensure compatibility with the ray transform operations and to optimize memory usage while maintaining sufficient precision for the computations.
        for t in range(self.time_bins):
            y_t = self.ray_trafo_module[t](vol[:, t])  # Apply the ray transform for time bin t to the corresponding volume at time bin t
            y[:, t] = y_t
        return y
    
    def backproject(self, sino: torch.Tensor) -> torch.Tensor:

        assert sino.ndim == 5, f"Expected input sinogram to have 5 dimensions (B, T, num_projections_per_time_bin, detector_shape, z), but got {sino.ndim} dimensions."
        assert sino.shape[1] == self.time_bins, f"Expected time dimension of input sinogram to match the number of time bins {self.time_bins}, but got {sino.shape[1]} time bins."

        # Allocate memory for the output reconstruction. The shape of the output reconstruction will be (B, T, D, H, W), where D, H, W are determined by the reconstruction space of the ray transform for each time bin.

        x = self._initialize_recon(batch_size=sino.shape[0], device=sino.device) # Initialize the reconstruction tensor with the appropriate shape and data type. The shape of the reconstruction will be (B, T, D, H, W), where D, H, W are determined by the reconstruction space of the ray transform for each time bin. The data type is set to float32 to ensure compatibility with the ray transform operations and to optimize memory usage while maintaining sufficient precision for the computations.

        for t in range(self.time_bins):
            x_t = self.adjoint_trafo_module[t](sino[:, t])  # Apply the adjoint transform for time bin t to the corresponding sinogram at time bin t
            x[:, t] = x_t

        return x
    
    
    def fbp_reconstruct(self, sino: torch.Tensor) -> torch.Tensor:

        assert sino.ndim == 5, f"Expected input sinogram to have 5 dimensions (B, T, num_projections_per_time_bin, detector_shape, z), but got {sino.ndim} dimensions."
        assert sino.shape[1] == self.time_bins, f"Expected time dimension of input sinogram to match the number of time bins {self.time_bins}, but got {sino.shape[1]} time bins."

        recon = []

        for t in range(self.time_bins):
            recon_t = self.fbp_trafo_module[t](sino[:, t])  # Apply the FBP transform for time bin t to the corresponding sinogram at time bin t
            recon.append(recon_t)

        recon = torch.stack(recon, dim=1)  # Stack the reconstructions for each time bin along the time dimension

        return recon
    
