import odl
import odl.applications.tomo as tomo
from odl.contrib.torch import OperatorModule
from odl.applications.tomo.analytic import filtered_back_projection as _fbp_module

import torch
import torch.nn.functional as F

from typing import Optional, Sequence, Tuple, overload


# NOTE: the installed odl version's `_fbp_filter` (odl/applications/tomo/analytic/filtered_back_projection.py)
# has two independent bugs that make `tomo.fbp_op` unusable with either `filter_type='Ram-Lak'` or a callable
# `filter_type`, confirmed by reading its source:
#   1. `filter_type, filter_type_in = str(filter_type).lower(), filter_type` unconditionally overwrites
#      `filter_type` with its stringified form *before* `if callable(filter_type):` is checked, so that
#      branch is dead code -- passing a callable never actually reaches it.
#   2. The `elif filter_type == 'ram-lak': pass` branch never assigns `filt`, so every 'Ram-Lak' call raises
#      `UnboundLocalError: cannot access local variable 'filt'`.
# This patch fixes both, checking callability first and setting `filt = norm_freq` for Ram-Lak (which is,
# mathematically, just the frequency-scaling-indicator-masked ramp/identity filter). Everything else is an
# unmodified copy of the original function.
def _fbp_filter_patched(norm_freq, filter_type, frequency_scaling):
    import numpy as np
    from odl.core.discr.discr_utils import get_array_and_backend

    filter_type_in = filter_type
    norm_freq, backend = get_array_and_backend(norm_freq)
    array_namespace = backend.array_namespace

    if callable(filter_type_in):
        filt = filter_type_in(norm_freq)
    else:
        filter_type = str(filter_type_in).lower()
        if filter_type == 'ram-lak':
            filt = norm_freq
        elif filter_type == 'shepp-logan':
            filt = norm_freq * array_namespace.sinc(norm_freq / (2 * frequency_scaling))
        elif filter_type == 'cosine':
            filt = norm_freq * array_namespace.cos(norm_freq * np.pi / (2 * frequency_scaling))
        elif filter_type == 'hamming':
            filt = norm_freq * (
                0.54 + 0.46 * array_namespace.cos(norm_freq * np.pi / (frequency_scaling)))
        elif filter_type == 'hann':
            filt = norm_freq * (
                array_namespace.cos(norm_freq * np.pi / (2 * frequency_scaling)) ** 2)
        else:
            raise ValueError('unknown `filter_type` ({})'.format(filter_type_in))

    indicator = (norm_freq <= frequency_scaling)
    filt *= indicator
    return filt


_fbp_module._fbp_filter = _fbp_filter_patched


GEOMETRIES = {"2D": {'Parallel2dGeometry', 'FanBeamGeometry'},
              "3D": {'Parallel3dAxisGeometry', 'ConeBeamGeometry'}}

PARALLEL_GEOMETRIES = {'Parallel2dGeometry', 'Parallel3dAxisGeometry'}

DEFAULT_CONFIG = {
    "geometry": "ConeBeamGeometry",
    "nDetectorCols": 512,
    "nDetectorRows": 96,
    "DetectorRowExtent": 80.0, # in mm
    "DetectorColExtent": 908.8, # in mm
    "GantrySpeed": 2 * torch.pi, # in radians per time unit
    "nViews": 512,
    "Flux": 10e14,
    "source_radius": 540,
    "det_radius": 950,  
    "det_curvature_radius": (950, torch.inf),
    "pitch": 0.0,
}




class RayTransform:
    """
    RayTransform class for tomographic reconstruction.

    This class encapsulates the configuration and initialization of a ray transform operator,
    supporting both 2D and 3D geometries, parallel and non-parallel setups, and various detector
    and object parameters.
    """
    def __init__(
        self,
        # Detector geometry parameters
        geometry: str,                                                                      # From odl.applications.tomo geometry options
        nDetectorCols: int,                                                                 # Number of detector columns            
        DetectorColExtent: float,                                                           # Physical extent for the detector columns.
        GantrySpeed: float,                                                                 # radians per time unit
        nViews: int,                                                                        # views per time unit
        Flux: float,                                                                        # number of photons per view
        # Required for 3D geometries
        nDetectorRows: int | None,                                                          # Number of detector rows. None for 2D geometries
        DetectorRowExtent: float | None,                                                    # Physical extent for the detector rows. None for 2D geometries
        rotAxis: Tuple[float, float, float] | None,                                         # Axis of rotation in 3D space
        # Required for non-parallel geometries
        source_radius: float | None,                                                        # Distance from the source to the rotation axis
        det_radius: float | None,                                                           # Distance from the detector to the rotation axis
        # Object geometry parameters
        extent: Tuple[float, float, float] | Tuple[float, float] | None,                    # Extent of the object in 3D space
        shape: Tuple[int, int, int] | Tuple[int, int] | None,                               # Shape of the object in 3D space
        
        # Optional detector parameters
        det_curvature_radius: Tuple[float, float] |Tuple[float] | None=None,                # Curvature radius of the detector. Only applicable to the non-parallel detectors. For 2D, it is Tuple[float] and for 3D cylindrical detectors, it is Tuple[float, float].
        pitch: float =0.0,                                                                  # Pitch of the helical trajectory
        # Implementation
        impl='astra_cuda',
        # Verbose flag for controlling output during computation
        verbose: bool = False,
    ):  
        self.geometry = geometry
        try:
            self.geometry_class = getattr(tomo, geometry)
        except AttributeError:
            raise ValueError(f"Geometry class for {geometry} could not be found in tomo module.")


        self.nDetectorRows = nDetectorRows
        self.nDetectorCols = nDetectorCols
        self.DetectorRowExtent = DetectorRowExtent
        self.DetectorColExtent = DetectorColExtent
        self.GantrySpeed = GantrySpeed
        self.nViews = nViews
        self.Flux = Flux
        self.rotAxis = rotAxis

        # Non-parallel geometries require source and detector radii to be specified.

        self.source_radius = source_radius
        self.det_radius = det_radius

        # Optional detector parameters
        # Curvature radius of the detector. None for flat detectors. (r, None) or (r, inf) gives a cylindrical detector. 

        self.det_curvature_radius = det_curvature_radius
        self.pitch = pitch
        
        self.extent = extent
        self.shape = shape

        self.impl = impl

        self.verbose = verbose

        assert any(geometry in geos for geos in GEOMETRIES.values()), f"Geometry {self.geometry} is not defined. It must be either a 2D geometry from {GEOMETRIES['2D']} or a 3D geometry from {GEOMETRIES['3D']}."
        assert self.geometry_class is not None, f"Geometry class for {self.geometry} could not be found in tomo module."
            

        # Detector columns parameters must be specified for all geometries

        assert DetectorColExtent is not None, "Physical extent for the detector columns must be specified."
        assert nDetectorCols is not None, "Number of detector columns must be specified."

        # NOTE: Detector rows parameters and axis of rotation must be specified for 3D geometries.
        #
        # Detector partition axis order is (col, row), i.e. `dpart`'s *first* axis is the in-plane
        # (fan/azimuthal) direction and its *second* axis is the direction physically aligned with
        # `rotAxis`, NOT (row, col) as might be assumed from `nDetectorRows`/`nDetectorCols` naming order.
        # Confirmed empirically: for a Parallel3dAxisGeometry/ConeBeamGeometry built with `dpart` shape
        # [n0, n1], the reco-space axis passed as `rotAxis` (e.g. D for a (D, H, W)-shaped volume with
        # `rotAxis=(1, 0, 0)`) lands 1:1 on the *second* projection-output axis, and the two axes
        # orthogonal to `rotAxis` jointly determine the first, regardless of which physical axis is chosen
        # as `rotAxis` -- this is odl's own `Geometry.det_axes_init` convention (det_axes[0] is the
        # in-plane axis, det_axes[1] is along `axis`), not something specific to this class. So a forward
        # projection's last two axes come out as (nDetectorCols, nDetectorRows), and the sinogram/projection
        # tensors this class returns follow that order throughout -- *not* (nDetectorRows, nDetectorCols).

        self.dim  = 2 if self.geometry in GEOMETRIES['2D'] else 3
        self.parallel = self.geometry in PARALLEL_GEOMETRIES
    
        if not self.parallel:
            assert source_radius is not None, "Source radius must be specified for non-parallel geometries."
            assert det_radius is not None, "Detector radius must be specified for non-parallel geometries."

        if self.dim == 2:

            self.nDetectorRows = None
            self.DetectorRowExtent = None

            self.detector_partition = odl.uniform_partition(min_pt=-DetectorColExtent/2, max_pt=DetectorColExtent/2, shape=nDetectorCols)

        if self.dim == 3:
            assert rotAxis is not None, "Rotation axis must be specified for 3D geometries."
            assert nDetectorRows is not None, "Number of detector rows must be specified for 3D geometries."
            assert DetectorRowExtent is not None, "Physical extent for the detector rows must be specified for 3D geometries."

            self.detector_partition = odl.uniform_partition(
                min_pt=[-DetectorColExtent/2, -DetectorRowExtent/2],
                max_pt=[DetectorColExtent/2, DetectorRowExtent/2],
                shape=[nDetectorCols, nDetectorRows]
            )

    def _init_reco_space(self, shape: tuple[int, ...], extent: tuple[float, ...]) -> odl.DiscretizedSpace:
        # dtype must be float32: astra (both astra_cpu and astra_cuda) requires float32 data, and the
        # projection space's dtype is inferred from this reco_space's dtype by odl's RayTransform. The
        # default dtype of odl.uniform_discr is float64, which fails at call time with
        # "ValueError: Failed to link dlpack array: Data must be float32" once a real backend runs.
        self.reco_space = odl.uniform_discr(
            min_pt=[-e/2 for e in extent],
            max_pt=[e/2 for e in extent],
            shape=shape,
            dtype='float32',
        )
        return self.reco_space


    @overload
    def _init_geometry(self, time_steps: float) -> tomo.Geometry: ...
    @overload
    def _init_geometry(self, time_steps: list[float]) -> list[tomo.Geometry]: ...

    def _init_geometry(self, time_steps: list[float]|float)->tomo.Geometry|list[tomo.Geometry]:
        """
        Initialize the geometry based on the given time steps.

        Parameters
        ----------
        time_steps : list[float] | float
            The time steps for the dynamic acquisition.

        Returns
        -------
        tomo.Geometry | list[tomo.Geometry]
            The initialized geometry or a list of geometries for each time step. If a single time step is provided, a single geometry is returned; otherwise, a list of geometries is returned.
        """
        def _get_geometry_by_case(angular_discr)->tomo.Geometry:


            if self.parallel:

                if self.dim == 3: #3DParallel
                    return self.geometry_class(apart=angular_discr, dpart=self.detector_partition, axis=self.rotAxis)
                else:             #2DParallel
                    return self.geometry_class(apart=angular_discr, dpart=self.detector_partition)
            else: 
                if self.dim == 3: #3DCone
                    assert self.det_curvature_radius is None or len(self.det_curvature_radius) == 2, "For 3D cone-beam geometry, det_curvature_radius must be None or a tuple of length 2."
                    return self.geometry_class(apart=angular_discr, dpart=self.detector_partition, src_radius=self.source_radius, det_radius=self.det_radius, det_curvature_radius=self.det_curvature_radius, pitch=self.pitch, axis=self.rotAxis)
                else:             #2DCone
                    assert self.det_curvature_radius is None or len(self.det_curvature_radius) == 1, "For 2D cone-beam geometry, det_curvature_radius must be None or a tuple of length 1."
                    return self.geometry_class(apart=angular_discr, dpart=self.detector_partition, src_radius=self.source_radius, det_radius=self.det_radius, det_curvature_radius=self.det_curvature_radius)

        assert isinstance(time_steps, float) or (isinstance(time_steps, list) and all(isinstance(t, float) for t in time_steps)), "time_steps must be a list of floats or a single float."

        if isinstance(time_steps, float):
            time_steps = [time_steps]
        self.time_steps = time_steps

        start_angles = torch.cumsum(torch.tensor([0] + time_steps[:-1]), dim=0) * self.GantrySpeed          # in radians    
        stop_angles= torch.cumsum(torch.tensor(time_steps), dim=0) * self.GantrySpeed                       # in radians

        views = [t * self.nViews for t in time_steps]

        self.angular_discretizations = [odl.uniform_partition(min_pt=float(theta_init), max_pt=float(theta_final), shape=int(v)) for theta_init, theta_final, v in zip(start_angles, stop_angles, views)]

        geometries = [_get_geometry_by_case(angular_discr) for angular_discr in self.angular_discretizations]

        return geometries if len(geometries) > 1 else geometries[0]

    def _init_data_params(self, shape: tuple[int, ...], extent: tuple[float, ...], time_steps: list[float]|float)->None:
        """Initialize the reconstruction space based on the given shape, extent, and time steps. It also builds the dynamic ray transform by setting up the views for each time step.
        input: 
            shape (tuple[int, ...]): The shape of the reconstruction space.
            extent (tuple[float, ...]): The physical extent of the reconstruction space.
            time_steps (list[float]|float): The list of time steps for the dynamic ray transform in units of time. If a single float is provided, it is interpreted as a single time step.
        
        
        """

        assert isinstance(time_steps, float) or (isinstance(time_steps, list) and all(isinstance(t, float) for t in time_steps)), "time_steps must be a list of floats or a single float."
        assert len(shape) == self.dim and len(extent) == self.dim, "Shape and extent must match the dimensionality of the reconstruction space."

        self.shape = shape
        self.extent = extent

        self.reco_space = self._init_reco_space(shape, extent)
        self.geometries = self._init_geometry(time_steps)

        if self.verbose:
            print("Reconstruction space initialized with shape:", self.shape, "and extent:", self.extent)
            print(f"Initialized {self.dim}D {"parallel" if self.parallel else ""}{self.geometry} with {len(self.geometries) if isinstance(self.geometries, list) else 1} geometries.")

    def _init_ray_transform(self, shape: tuple[int, ...], extent: tuple[float, ...], time_steps: list[float]|float) -> None:
        """
        Initialize the ray transform operator.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the reconstruction space.
        extent : tuple[float, ...]
            The physical extent of the reconstruction space.
        time_steps : list[float] | float
            The time steps for the dynamic acquisition.
        """
        assert isinstance(time_steps, float) or (isinstance(time_steps, list) and all(isinstance(t, float) for t in time_steps)), "time_steps must be a list of floats or a single float."

        self._init_data_params(shape, extent, time_steps)
        # NOTE: We want to wrap each RayTransform in an OperatorModule to ensure consistent handling of operators and automatic differentiation.
        self.ray_transform = [OperatorModule(tomo.RayTransform(self.reco_space, geometry, impl=self.impl)) for geometry in (self.geometries if isinstance(self.geometries, list) else [self.geometries])]  if isinstance(time_steps, list) else OperatorModule(tomo.RayTransform(self.reco_space, self.geometries, impl=self.impl))

        # NOTE: OperatorModule only exposes `forward` (it's a plain nn.Module wrapping one odl Operator) --
        # it has no `.adjoint`. The adjoint of the *underlying* odl operator (`.operator.adjoint`) is itself
        # a distinct odl Operator, so it needs its own OperatorModule to be callable on torch tensors.
        if isinstance(self.ray_transform, list):
            self.ray_transform_adjoint = [OperatorModule(rt.operator.adjoint) for rt in self.ray_transform]
        else:
            self.ray_transform_adjoint = OperatorModule(self.ray_transform.operator.adjoint)

        # NOTE: tomo.fbp_op needs the raw odl Operator (it reads .domain/.range/.adjoint etc., which
        # OperatorModule -- a plain nn.Module wrapper -- does not expose), and its result is itself a raw
        # odl Operator that needs its own OperatorModule to be callable on torch tensors, same as adjoint
        # above. (See the `_fbp_filter_patched` monkeypatch at the top of this module for why plain
        # `filter_type='Ram-Lak'` needed a fix before it could be used here at all.)
        if isinstance(self.ray_transform, list):
            self.fbp_operator = [
                OperatorModule(tomo.fbp_op(rt.operator, filter_type='Ram-Lak', frequency_scaling=1.0))
                for rt in self.ray_transform
            ]
        else:
            self.fbp_operator = OperatorModule(
                tomo.fbp_op(self.ray_transform.operator, filter_type='Ram-Lak', frequency_scaling=1.0)
            )


    @overload
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def __call__(self, x) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(self.ray_transform, list):
            assert isinstance(x, list) and len(x) == len(self.ray_transform), "Input must be a list of tensors when the ray transform is a list."
            return [rt(xi) for xi, rt in zip(x, self.ray_transform)]
        return self.ray_transform(x)

    @overload
    def adjoint(self, y: torch.Tensor) -> torch.Tensor: ...

    @overload
    def adjoint(self, y: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def adjoint(self, y) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(self.ray_transform_adjoint, list):
            assert isinstance(y, list) and len(y) == len(self.ray_transform_adjoint), "Input must be a list of tensors when the ray transform adjoint is a list."
            return [rt_adj(yi) for yi, rt_adj in zip(y, self.ray_transform_adjoint)]
        return self.ray_transform_adjoint(y)

    @overload
    def FBP(self, y: torch.Tensor) -> torch.Tensor: ...

    @overload
    def FBP(self, y: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def FBP(self, y) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(self.fbp_operator, list):
            assert isinstance(y, list) and len(y) == len(self.fbp_operator), "Input must be a list of tensors when the FBP operator is a list."
            return [fbp(yi) for yi, fbp in zip(y, self.fbp_operator)]
        return self.fbp_operator(y)

    def simulate_noise(
            self, x: torch.Tensor | list[torch.Tensor], 
            mu_water: float=0.0195, # in mm^-1 at 70 keV. See https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html for reference.
            epsilon: float=1e-6  # Small constant to avoid division by zero or log of zero.
                       ) -> torch.Tensor | list[torch.Tensor]:


        """
        Simulate noise in the input tensor. This achieved by transforming the input data x (HU) to attenuation coefficients. The data is then forward projected using the ray transform, and noise is added to the projections to simulate noisy measurments. 

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            The input image in Hounsfield units (HU). If a list of tensors is provided, noise will be simulated for each tensor individually.
        mu_water : float, optional
            The linear attenuation coefficient of water at the relevant X-ray energy (in mm^-1). Default is 0.0195 (at 70 keV).
        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            The noisy projections after simulating CT Poisson noise.
        """
        
        
        x = (x / 1000.0 + 1.0) * mu_water if isinstance(x, torch.Tensor) else [(xi / 1000.0 + 1.0) * mu_water for xi in x]   # Convert HU to attenuation coefficients
        projections = self(x)

        # Simulate Lambert-Beer law for X-ray attenuation

        areafraction = 1 / (self.nDetectorCols * self.nDetectorRows) if isinstance(self.nDetectorCols, int) and isinstance(self.nDetectorRows, int) else 1 /self.nDetectorCols # Compute the area fraction of each detector element compared to the total detector area
        N0 = self.Flux * areafraction / self.nViews # Initial number of photons per detector element per view
        counts = N0 * torch.exp(-projections) if isinstance(projections, torch.Tensor) else [N0 * torch.exp(-pi) for pi in projections]

        noisy_counts = torch.poisson(counts) if isinstance(counts, torch.Tensor) else [torch.poisson(ci) for ci in counts]

        noisy_projections = -torch.log((noisy_counts + epsilon) / N0) if isinstance(noisy_counts, torch.Tensor) else [-torch.log((nc + epsilon) / N0) for nc in noisy_counts]

        return noisy_projections

      
 