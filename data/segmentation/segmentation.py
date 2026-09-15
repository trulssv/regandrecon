from totalsegmentator.python_api import totalsegmentator
from typing import Union
from nibabel import Nifti1Image
import nibabel as nib
from pathlib import Path
import os
import torch
import numpy as np
import cc3d


# Support both:
#  - package import: `from data_processing.segmentation.segmentation import Segmenter`
#  - direct script execution: `python data_processing/segmentation/segmentation.py`
try:
    from .utils import *  # type: ignore
except ImportError:  # pragma: no cover
    from utils import *  # type: ignore

from data.segmentation.arguments.segmentator_arguments import SegmentationArguments



class Segmenter:
    """
    Segmenter class for handling medical image segmentation tasks.
    This class provides an interface for segmenting medical images using various predefined tasks.
    It supports setting input images, specifying output directories, and saving segmentation results.
    Attributes
    ----------
    input : Union[str, Path, Nifti1Image], optional
        The input image to be segmented. Can be a file path, Path object, or a Nifti1Image.
    output_dir : str or Path, optional
        Directory where segmentation outputs will be saved.
    license_number : str, optional
        License number required for segmentation tools, if applicable.
    Methods
    -------
    set_output_dir(output_dir)
        Set the directory where segmentation outputs will be saved.
    set_input(input)
        Set the input image for segmentation.
    segment_multitasks(tasks=['total', 'tissue_4_types', 'body'], fast=False, save_segmentations=True)
        Perform segmentation for multiple tasks. Optionally saves the segmentation results.
    segment_subtask(task='total', fast=False)
        Perform segmentation for a single specified task.
    save_seg(seg, path)
        Save a segmentation result to the specified file path.
    """

    
    def __init__(self, output_dir=None, license_number=None, device: str = "gpu", nr_thr_resamp: int = 1, nr_thr_saving: int = 2, robust_crop: bool = True):
        self.input = None
        self.output_dir = output_dir
        self.license_number = license_number
        self.device = device
        self.nr_thr_resamp = int(nr_thr_resamp)
        self.nr_thr_saving = int(nr_thr_saving)
        self.robust_crop = robust_crop
        
    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def set_input(self, input: Union[str, Path, Nifti1Image]):
        self.input = input

    def segment_multitasks(self, tasks=['total', 'tissue_4_types', 'body'], fast=False, save_segmentations=True):
        assert self.input is not None, 'Input is not set'
        subsegmentations = []
        for task in tasks:

            if task == 'air':
                subsegmentations.append(AirSegmenter(self.input))
            elif task == 'bone':
                subsegmentations.append(BoneSegmenter(self.input))
            else:
                subsegmentations.append(self.segment_subtask(task=task, fast=fast))

            if save_segmentations:
                self.save_seg(subsegmentations[-1], Path(self.output_dir, f'{task}.nii.gz'))

        return subsegmentations

    def segment_subtask(self, task='total', fast=False): # total, tissue_4_types, body
        assert self.input is not None, 'Input is not set'

        # Reduce RAM pressure by limiting background workers where possible.
        seg = totalsegmentator(
            self.input,
            task=task,
            fast=fast,
            license_number=self.license_number,
            device=self.device,
            nr_thr_resamp=self.nr_thr_resamp,
            nr_thr_saving=self.nr_thr_saving,
            robust_crop=self.robust_crop,
        )
        return seg
    
    def save_seg(self, seg, path):
        path = Path(path)
        print(f"Saving segmentation to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to a temp file in the same directory then rename.
        # This protects long runs from leaving corrupted .nii.gz on interruptions.
        # IMPORTANT: nibabel infers file type from the filename extension.
        # Using "air.nii.gz.tmp" breaks this inference; keep the .nii.gz suffix.
        tmp_path = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz") if path.name.endswith(".nii.gz") else (path.name + ".tmp"))
        nib.save(seg, str(tmp_path))
        os.replace(str(tmp_path), str(path))




class ModifiedSegmenter(Segmenter):
    def __init__(
            self, 
            data_dir=None, 
            device: str = "gpu", 
            nr_thr_resamp: int = 1, 
            nr_thr_saving: int = 2,
            preprocess_cfg: dict = None,
            segmenter_args: dict = SegmentationArguments              
):
        super().__init__(output_dir=data_dir, license_number=segmenter_args.license_number, device=device, nr_thr_resamp=nr_thr_resamp, nr_thr_saving=nr_thr_saving)

        self.preprocess_cfg = preprocess_cfg
        self.data_dir = data_dir
        self.segmenter_args = segmenter_args




    def prepare_nifti(self, x: torch.Tensor, meta_data: dict = None) -> tuple[Path, torch.Size]:
            

        if not isinstance(self.data_dir, Path):
            path_to_data = Path(self.data_dir) if self.data_dir is not None else Path(path_to_data)

        else:
            path_to_data = self.data_dir


        nifti_path = path_to_data / "volume.nii.gz"

        assert x.ndim == 3, f"Expected data to have 3 dimensions (D, H, W), but got {x.ndim} dimensions. Please check the data format."

        HU_range = self.preprocess_cfg.get("HU_RANGE", [-1024, 3072])

        # Rescale from [0, 1] to HU range

        x = x * (HU_range[1] - HU_range[0]) + HU_range[0]

        # Save as nifti
        affine = self._get_affine(meta_data)
        nib.save(nib.Nifti1Image(x, affine), str(nifti_path))
        return nifti_path


    def _get_affine(self, meta_data: dict) -> np.ndarray:
        
        assert "resampled_pixel_spacing" in meta_data, "Expected 'resampled_pixel_spacing' in meta_data for correct affine construction, but it is not present. Please check the meta_data format."
        assert "resampled_slice_thickness" in meta_data, "Expected 'resampled_slice_thickness' in meta_data for correct affine construction, but it is not present. Please check the meta_data format."

        pixel_spacing = meta_data["resampled_pixel_spacing"]
        slice_thickness = meta_data["resampled_slice_thickness"]

        s_x, s_y, s_z = pixel_spacing[0].item(), pixel_spacing[1].item(), slice_thickness.item()

      

        affine = np.diag([s_z, s_y, s_x, 1.0])  # Note the order of s_z, s_y, s_x to match (D, H, W) -> (z, y, x)

        return affine


    def save_segmentations(self, segmentations: dict[str, torch.Tensor], meta_data: dict = None):
        for task, seg in segmentations.items():
            torch.save(seg, Path(self.output_dir, f'{task}.pt'))


    def  __call__(self, x: torch.Tensor, meta_data: dict = None, data_dir: str=None) -> dict[str, torch.Tensor]:
        if data_dir is not None:
            self.data_dir = data_dir
        self.set_output_dir(self.data_dir)
        

        segmentations = {}

        for task in self.segmenter_args.tasks:
            segmentations[task] = torch.zeros_like(x, dtype=torch.int16)  # Initialize segmentation tensor for each task with the same shape as input x, but with integer type for segmentation labels.


        assert x.ndim == 5, f"Expected data to have 5 dimensions (B, T, D, H, W), but got {x.ndim} dimensions. Please check the data format."
        b, t, d, h, w = x.shape

        x = x.cpu().numpy()  # Convert to numpy array for easier handling with nibabel and totalsegmentator

        for i in range(b):
            for j in range(t):
                print(f"Processing batch {i+1}/{b}, time bin {j+1}/{t}...")

                x_ij = x[i, j] # shape (D, H, W)

                nifti_path = self.prepare_nifti(x_ij, meta_data=meta_data)
                self.set_input(nifti_path)

                segmentations_ij = self.segment_multitasks(
                                                        tasks=self.segmenter_args.tasks, 
                                                        fast=self.segmenter_args.mode, 
                                                        save_segmentations=False
                                                        )
                for task in self.segmenter_args.tasks:

                    task_seg_ij = segmentations_ij[self.segmenter_args.tasks.index(task)]

                    # Convert to torch tensor

                    task_seg_ij = torch.from_numpy(np.array(task_seg_ij.dataobj)).to(torch.int16)

                    ## ... Post-process

                    task_seg_ij = self.post_process_segmentation(task_seg_ij)

                    



                    segmentations[task][i, j] = task_seg_ij  # shape (D, H, W)

        
        return segmentations

    def post_process_segmentation(self, seg: torch.Tensor) -> torch.Tensor:

        seg[seg > 0] = 1
        
        # Connected component analysis to keep only the 2 largest connected components
        seg_np = seg.cpu().numpy()
        labels, num_labels = cc3d.connected_components(seg_np, connectivity=26, binary_image=True, return_N=True)
        if num_labels > 2:
            largest_labels = np.argsort(np.bincount(labels.flat)[1:])[-2:] + 1
            seg_np[~np.isin(labels, largest_labels)] = 0
        seg = torch.from_numpy(seg_np).to(seg.device)
        return seg