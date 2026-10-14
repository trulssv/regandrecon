import astra
import odl
import odl.applications.tomo as tomo
from odl.contrib.torch import OperatorModule
from odl.applications.tomo.analytic import filtered_back_projection as _fbp_module

import torch
import torch.nn.functional as F

from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence, Tuple, List, cast, overload


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
    from odl.core.array_API_support import get_array_and_backend

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


# NOTE: OperatorModule evaluates the operator on its domain's device, which is the CPU for our numpy-backed odl spaces. Every call on a
# CUDA tensor therefore copies the data to the host and back, around ASTRA's own host <-> GPU copies, and synchronizes with the host:
# about 50 ms per forward or back projection of a 50x256x256 volume, of which ASTRA's kernels take 3-5 ms. (odl spaces with
# impl='pytorch' are no faster, as odl copies and checks the data instead.) AstraCudaModule runs ASTRA directly on the CUDA tensors,
# reusing the projector and scaling factors of odl's own backend, so that it computes exactly what OperatorModule computes.

_astra_gpu_index: int | None = None     # the GPU ASTRA was last set to run on, see _set_astra_gpu


def _set_astra_gpu(device: torch.device) -> None:
    """ASTRA runs on one GPU per process (GPU 0 unless set), which must be the one holding the linked tensors."""
    global _astra_gpu_index
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index != _astra_gpu_index:
        astra.set_gpu_index(index)
        _astra_gpu_index = index


def _astra_project(impl, x: torch.Tensor, adjoint: bool) -> torch.Tensor:
    """ASTRA forward (adjoint=False) or back projection of a single CUDA volume or sinogram, in odl's layout and scaling (see
    AstraCudaImpl.call_forward/call_backward). Sinograms are (angles, *detector) in odl and (det rows, angles, det cols) in ASTRA."""
    _set_astra_gpu(x.device)
    # NOTE: ASTRA's calls block until its kernels are done, but it does not know about torch's stream, so the work torch queued to
    # produce x must be finished first.
    torch.cuda.current_stream(x.device).synchronize()
    with impl._mutex:
        if adjoint:
            proj = x.permute(*impl.transpose_tuple).contiguous()
            vol = torch.empty(astra.geom_size(impl.vol_geom), dtype=torch.float32, device=x.device)
            astra.experimental.direct_BP3D(impl.projector_id, vol, proj)
            return vol * impl.bp_scaling_factor
        vol = x.contiguous()
        proj = torch.empty(astra.geom_size(impl.proj_geom), dtype=torch.float32, device=x.device)
        astra.experimental.direct_FP3D(impl.projector_id, vol, proj)
        return (proj * impl.fp_scaling_factor).permute(*impl.transpose_tuple).contiguous()


class _AstraProjection(torch.autograd.Function):
    """Forward (adjoint=False) or back projection with the same gradients as odl's OperatorFunction: the adjoint projection scaled by
    the ratio of the domain and range cell volumes, which is the transpose with respect to the plain (unweighted) dot product."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, impl, adjoint: bool, in_shape: tuple[int, ...], grad_scale: float) -> torch.Tensor:
        ctx.impl, ctx.adjoint, ctx.grad_scale = impl, adjoint, grad_scale
        extra_shape = x.shape[:-len(in_shape)]
        ctx.extra_shape = extra_shape
        out = [_astra_project(impl, xi, adjoint) for xi in x.detach().reshape(-1, *in_shape)]
        return torch.stack(out).reshape(*extra_shape, *out[0].shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        out_shape = grad_out.shape[len(ctx.extra_shape):]
        grad = [_astra_project(ctx.impl, gi, not ctx.adjoint) for gi in grad_out.detach().reshape(-1, *out_shape)]
        grad = torch.stack(grad).reshape(*ctx.extra_shape, *grad[0].shape) * ctx.grad_scale
        return grad, None, None, None, None


class AstraCudaModule(torch.nn.Module):
    """Drop-in replacement of OperatorModule(ray_trafo) (adjoint=False) or OperatorModule(ray_trafo.adjoint) (adjoint=True) for an
    odl astra_cuda RayTransform of a 3D geometry, which runs ASTRA directly on CUDA tensors (see the note above). CPU tensors are
    passed to OperatorModule. Like OperatorModule, it accepts extra leading (batch) axes and exposes the odl operator as .operator."""

    def __init__(self, ray_trafo: tomo.RayTransform, adjoint: bool = False) -> None:
        super().__init__()
        self.ray_trafo = ray_trafo
        self.adjoint = adjoint
        self.operator = ray_trafo.adjoint if adjoint else ray_trafo
        self._odl_module = OperatorModule(self.operator)
        domain, range_ = cast(Any, self.operator.domain), cast(Any, self.operator.range)
        # NOTE: the gradient scale of OperatorFunction, i.e. the domain over the range cell volume of the operator.
        self.grad_scale = domain.weighting.const / range_.weighting.const
        self.in_shape = tuple(domain.shape)

    @staticmethod
    def supports(ray_trafo: tomo.RayTransform) -> bool:
        domain = cast(Any, ray_trafo.domain)
        return ray_trafo.impl == "astra_cuda" and domain.ndim == 3 and domain.impl == "numpy"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self._odl_module(x)
        assert x.dtype == torch.float32, f"ASTRA needs float32 data, got {x.dtype}."
        assert tuple(x.shape[-len(self.in_shape):]) == self.in_shape, f"Expected input of shape (*, {self.in_shape}), got {tuple(x.shape)}."
        return _AstraProjection.apply(x, self.ray_trafo.get_impl(True), self.adjoint, self.in_shape, self.grad_scale)


def _projection_module(ray_trafo: tomo.RayTransform, adjoint: bool = False) -> torch.nn.Module:
    """AstraCudaModule where it applies, otherwise odl's OperatorModule."""
    if AstraCudaModule.supports(ray_trafo):
        return AstraCudaModule(ray_trafo, adjoint)
    return OperatorModule(ray_trafo.adjoint if adjoint else ray_trafo)


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
        Flux: float,                                                                        # number of photons per time unit
        # Required for 3D geometries
        nDetectorRows: int | None,                                                          # Number of detector rows. None for 2D geometries
        DetectorRowExtent: float | None,                                                    # Physical extent for the detector rows. None for 2D geometries
        rotAxis: Tuple[float, ...] | None,                                                  # Axis of rotation in 3D space
        # Required for non-parallel geometries
        source_radius: float | None,                                                        # Distance from the source to the rotation axis
        det_radius: float | None,                                                           # Distance from the detector to the rotation axis
        # Object geometry parameters
        extent: Tuple[float, ...] | None,                                                   # Extent of the object in 3D space
        shape: Tuple[int, ...] | None,                                                      # Shape of the object in 3D space
        
        # Optional detector parameters
        det_curvature_radius: Tuple[float, ...] | None=None,                                # Curvature radius of the detector. Only applicable to the non-parallel detectors. For 2D, it is Tuple[float] and for 3D cylindrical detectors, it is Tuple[float, float].
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

        # Operators of the most recently used geometries, see _init_ray_transform.
        self._geometry_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._geometry_key: tuple | None = None

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

    # Attributes that depend on the geometry, saved and restored by the geometry cache of _init_ray_transform.
    _GEOMETRY_ATTRS = ("shape", "extent", "time_steps", "reco_space", "angular_discretizations", "geometries", "ray_transform", "ray_transform_adjoint")
    GEOMETRY_CACHE_SIZE = 8

    def _init_ray_transform(self, shape: tuple[int, ...], extent: tuple[float, ...], time_steps: list[float]|float) -> None:
        """
        Initialize the ray transform operator.

        The operators of the GEOMETRY_CACHE_SIZE most recently used geometries (shape, extent and time steps) are cached, so that
        switching between studies with the same geometry (e.g. HLPDModel.set_geometry for every training study) costs nothing.

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

        key = (tuple(shape), tuple(float(e) for e in extent), tuple(time_steps) if isinstance(time_steps, list) else time_steps)
        self._geometry_key = key
        if key in self._geometry_cache:
            self._geometry_cache.move_to_end(key)
            for name, value in self._geometry_cache[key].items():
                if name in self._GEOMETRY_ATTRS:
                    setattr(self, name, value)
            return

        self._init_data_params(shape, extent, time_steps)
        # NOTE: We want to wrap each RayTransform in a torch module (AstraCudaModule, or odl's OperatorModule where it does not apply) to ensure consistent handling of operators and automatic differentiation.
        self.ray_transform = [_projection_module(tomo.RayTransform(self.reco_space, geometry, impl=self.impl)) for geometry in (self.geometries if isinstance(self.geometries, list) else [self.geometries])]  if isinstance(time_steps, list) else _projection_module(tomo.RayTransform(self.reco_space, self.geometries, impl=self.impl))

        # NOTE: OperatorModule only exposes `forward` (it's a plain nn.Module wrapping one odl Operator) --
        # it has no `.adjoint`. The adjoint of the *underlying* odl operator (`.operator.adjoint`) is itself
        # a distinct odl Operator, so it needs its own OperatorModule to be callable on torch tensors.
        if isinstance(self.ray_transform, list):
            self.ray_transform_adjoint = [_projection_module(rt.operator, adjoint=True) for rt in self.ray_transform]
        else:
            self.ray_transform_adjoint = _projection_module(self.ray_transform.operator, adjoint=True)

        self._geometry_cache[key] = {name: getattr(self, name) for name in self._GEOMETRY_ATTRS} | {"fbp_operator": None}
        while len(self._geometry_cache) > self.GEOMETRY_CACHE_SIZE:
            self._geometry_cache.popitem(last=False)

    @property
    def fbp_operator(self) -> OperatorModule | list[OperatorModule]:
        """The FBP operator(s) of the current geometry. Built on first use, as they take about 0.4 GB of host memory per time bin
        and are not needed for training."""
        assert self._geometry_key is not None, "Must initialize the ray transform before using the FBP operator, see _init_ray_transform."
        entry = self._geometry_cache[self._geometry_key]
        if entry["fbp_operator"] is None:
            # NOTE: tomo.fbp_op needs the raw odl Operator (it reads .domain/.range/.adjoint etc., which
            # OperatorModule -- a plain nn.Module wrapper -- does not expose), and its result is itself a raw
            # odl Operator that needs its own OperatorModule to be callable on torch tensors, same as adjoint
            # above. (See the `_fbp_filter_patched` monkeypatch at the top of this module for why plain
            # `filter_type='Ram-Lak'` needed a fix before it could be used here at all.)
            if isinstance(self.ray_transform, list):
                entry["fbp_operator"] = [
                    OperatorModule(tomo.fbp_op(rt.operator, filter_type='Ram-Lak', frequency_scaling=1.0))
                    for rt in self.ray_transform
                ]
            else:
                entry["fbp_operator"] = OperatorModule(
                    tomo.fbp_op(self.ray_transform.operator, filter_type='Ram-Lak', frequency_scaling=1.0)
                )
        return entry["fbp_operator"]


    @overload
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def __call__(self, x) -> torch.Tensor | list[torch.Tensor]:
        """Maps a 3D volume or 4D spatio-temporal sequence of 3D volumes to projection data.
        
        Args:
            x: 3D volume B x D x H x W or a spatio-temporal 4D volume T x B x D x H x W where the first (temporal) dimension is represented as a list.

        Out:
            y: Projection data. Either B x nAngles x nCol x nRow for a 3D volume or T x B x nAngles x nCol x nRow for a spatiotemporal volume, where the first dimension is represented as a list. 
               (nCols, nRows) is the detector partition and nAngles is the angular (view) partition. 
        
        """
        if isinstance(x, list):
            assert isinstance(self.ray_transform, list) and len(x) == len(self.ray_transform), "Input must be a list of tensors when the ray transform is a list."
            return [rt(xi) for xi, rt in zip(x, self.ray_transform)]
        elif isinstance(x, torch.Tensor) and isinstance(self.ray_transform, torch.nn.Module):
            return self.ray_transform(x)
        else:
            raise ValueError("Input must be a list of tensors or a torch.Tensor compatible with the ray transform.")


        
    @overload
    def adjoint(self, y: torch.Tensor) -> torch.Tensor: ...

    @overload
    def adjoint(self, y: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def adjoint(self, y) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(y, list):
            assert isinstance(self.ray_transform_adjoint, list) and len(y) == len(self.ray_transform_adjoint), "Input must be a list of tensors when the ray transform adjoint is a list."
            return [rt_adj(yi) for yi, rt_adj in zip(y, self.ray_transform_adjoint)]
        elif isinstance(y, torch.Tensor) and isinstance(self.ray_transform_adjoint, torch.nn.Module):
            return self.ray_transform_adjoint(y)
        else:
            raise ValueError("Input must be a list of tensors or a torch.Tensor compatible with the ray transform adjoint.")
    

        

    @overload
    def FBP(self, y: torch.Tensor) -> torch.Tensor: ...

    @overload
    def FBP(self, y: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def FBP(self, y) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(y, list):
            assert isinstance(self.fbp_operator, list) and len(y) == len(self.fbp_operator), "Input must be a list of tensors when the FBP operator is a list."
            return [fbp(yi) for yi, fbp in zip(y, self.fbp_operator)]
        elif isinstance(y, torch.Tensor) and isinstance(self.fbp_operator, OperatorModule):
            return self.fbp_operator(y)
        else:
            raise ValueError("Input must be a list of tensors or a torch.Tensor compatible with the FBP operator.")

    def forward_bin(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Applies the ray transform of time bin t to a single 3D volume B x D x H x W."""
        assert isinstance(self.ray_transform, list), "A time dependent ray transform must be initialized to use time bin specific operators."
        return self.ray_transform[t](x)

    def adjoint_bin(self, y: torch.Tensor, t: int) -> torch.Tensor:
        """Applies the adjoint ray transform of time bin t to projection data B x nAngles x nCol x nRow."""
        assert isinstance(self.ray_transform_adjoint, list), "A time dependent ray transform adjoint must be initialized to use time bin specific operators."
        return self.ray_transform_adjoint[t](y)

    def fbp_bin(self, y: torch.Tensor, t: int) -> torch.Tensor:
        """Applies the FBP operator of time bin t to projection data B x nAngles x nCol x nRow."""
        assert isinstance(self.fbp_operator, list), "A time dependent FBP operator must be initialized to use time bin specific operators."
        return self.fbp_operator[t](y)

    @overload
    def add_poisson_noise(self, y: torch.Tensor, flux: float | None = None, epsilon: float = 1e-6) -> torch.Tensor: ...

    @overload
    def add_poisson_noise(self, y: list[torch.Tensor], flux: float | None = None, epsilon: float = 1e-6) -> list[torch.Tensor]: ...

    def add_poisson_noise(
            self, y: torch.Tensor | list[torch.Tensor],
            flux: float | None = None,  # Number of photons per view. Defaults to self.Flux.
            epsilon: float = 1e-6       # Small constant to avoid division by zero or log of zero.
                       ) -> torch.Tensor | list[torch.Tensor]:
        """
        Add CT Poisson noise to clean projections y, i.e. line integrals of attenuation values in mm^-1 as returned by self(x).
        See `add_poisson_noise` (module level) for the noise model.
        """
        n0 = photon_count(self.Flux if flux is None else flux, self.nViews, self.nDetectorCols, self.nDetectorRows)
        return add_poisson_noise(y, n0=n0, epsilon=epsilon)


def operator_norm(
        forward: Callable[[torch.Tensor], torch.Tensor],
        adjoint: Callable[[torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],                         # domain shape, including the batch dimension
        max_iters: int = 20,
        rtol: float = 1e-3,                             # stop when the estimate changes less than this, relative
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        ) -> float:
    """Estimates the operator norm ||A|| by power iteration on A^* A, e.g. for the ray transform of one time bin with
    forward = lambda x: ray_trafo.forward_bin(x, t). Uses its own generator, so that it does not consume the global RNG."""
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(shape, generator=generator).to(device)
    u = u / u.norm()
    norm_sq = 0.0
    with torch.no_grad():
        for _ in range(max_iters):
            w = adjoint(forward(u))
            new_norm_sq = w.norm().item()               # ||A^* A u|| with ||u|| = 1, converges to ||A||^2 from below
            u = w / new_norm_sq
            converged = abs(new_norm_sq - norm_sq) <= rtol * new_norm_sq
            norm_sq = new_norm_sq
            if converged:
                break
    return norm_sq ** 0.5


def photon_count(flux: float, nViews: int, nDetectorCols: int, nDetectorRows: int | None = None) -> float:
    """Initial number of photons N0 per detector element per view, given the total flux per view spread evenly over the detector."""
    n_detector_elements = nDetectorCols * (nDetectorRows if nDetectorRows is not None else 1)
    return flux / (n_detector_elements * nViews)


@overload
def add_poisson_noise(y: torch.Tensor, n0: float, epsilon: float = 1e-6) -> torch.Tensor: ...

@overload
def add_poisson_noise(y: list[torch.Tensor], n0: float, epsilon: float = 1e-6) -> list[torch.Tensor]: ...

def add_poisson_noise(y, n0: float, epsilon: float = 1e-6) -> torch.Tensor | list[torch.Tensor]:
    """
    Simulate noisy measurements from clean projections y (line integrals of attenuation values in mm^-1).
    Following the Lambert-Beer law, the expected photon counts are N0 * exp(-y). These are Poisson sampled and log-transformed back to line integrals.

    Parameters
    ----------
    y : torch.Tensor | list[torch.Tensor]
        The clean projections. If a list of tensors is provided, noise is added to each tensor individually.
    n0 : float
        The initial number of photons per detector element per view, see `photon_count`.
    epsilon : float, optional
        Small constant to avoid the log of zero for detector elements without counts.
    Returns
    -------
    torch.Tensor | list[torch.Tensor]
        The noisy projections.
    """
    if isinstance(y, list):
        return [add_poisson_noise(yi, n0=n0, epsilon=epsilon) for yi in y]


    # clamp y to remove negative values that could lead to invalid photon counts
    if torch.any(y < 0):
        print(f"Warning: {100 * torch.sum(y < 0) / y.numel():.2f}% negative values in y detected. Clamping to 0.")
        y = torch.clamp(y, min=0.0)

    
    noisy_counts = torch.poisson(n0 * torch.exp(-y))
    return -torch.log((noisy_counts + epsilon) / n0)
