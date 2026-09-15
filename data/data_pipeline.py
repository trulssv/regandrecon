from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import json

from data.preprocess_volume import VolumePreprocessor
from data.utils import _to_jsonable
from visualization.dynamic_visualization import DynamicVisualization
from data.denoising.denoising import Denoiser
from data.segmentation.segmentation import ModifiedSegmenter
from operators.ray_transform import DynamicRayTransform

from data.data_loaders import RegAndReconDataset
from data.segmentation.arguments.segmentator_arguments import SegmentationArguments

def visualize_planes(x: torch.Tensor, meta_data: dict, vmin, vmax, save_dir: Path, prefix: str)->None :
    vis = DynamicVisualization(x, meta_data, vmin=vmin, vmax=vmax, save_dir=save_dir)
    for plane in ["axial", "coronal", "sagittal"]:
        vis.visualize_plane_dynamic(plane=plane, slice_idx=None, prefix=prefix)
        vis.visualize_slices(plane=plane, time_bin=0, prefix=prefix)



def data_pipeline(steps: dict[str, bool]=None, mode: str="val")->None:
    if steps is None:
        steps = {}

    qualities = ["high"] 
    assert mode in ["train", "val", "test"], f"Invalid mode {mode}. Expected one of ['train', 'val', 'test']."
    data_root = Path("/mnt/data/LDDMM")

    dataset = RegAndReconDataset(qualities=qualities, mode=mode, data_root=data_root)


    with open("data/data.json", "r") as f:
        data_cfg = json.load(f)
    
    with open("data/preprocess.json", "r") as f:
        preprocess_cfg = json.load(f)

    with open("operators/ray_trafo.json", "r") as f:
        ray_trafo_cfg = json.load(f)
    

    # Create Loaders

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    

    ## ... Create Operators

    # Volume Preprocessor

    volume_preprocessor = VolumePreprocessor(preprocess_cfg) # Resizes and rescales the volume to [0, 1] range based on the HU range in the preprocess_cfg

    # Denoising

    denoiser = Denoiser()
    sigma = 0.0

    # Segmentation

    segmenter = ModifiedSegmenter(device="gpu", preprocess_cfg=preprocess_cfg, segmenter_args=SegmentationArguments)

   
    for i, batch in enumerate(tqdm(data_loader, desc="Processing Batches")):
        print(f"Batch {i}:")

        # Data pipline
        x = batch["volume"]  # b, t, d, h, w
        meta_data = batch["meta_data"]  # Get the meta data for the first (and only) sample in the batch. This can be customized based on the requirements of the project (e.g., handling multiple samples in a batch, etc.).
        study_dir = Path(batch["study_dir"][0])  # Get the study directory for the first (and only) sample in the batch. This can be useful for saving results, etc.



        # Ray Tramnsform

        ray_trafo = DynamicRayTransform(ray_trafo_cfg=ray_trafo_cfg, preprocess_cfg=preprocess_cfg, meta_data=meta_data)



        x = x.to("cuda:0")
        x = volume_preprocessor(x)

        if steps.get("denoising", True):

            x = denoiser(x, sigma=sigma)

        if steps.get("segmentation", True):

            print(f"Running segmentations with tasks {segmenter.segmenter_args.tasks} for study {study_dir}...")

            x_seg = segmenter(x, meta_data=meta_data, data_dir=study_dir)

            # segmenter return a dict with the segmentations tasks as keys and segmentations as values. Now, we just pick the first task for visualization and saving results. 
            # This can be customized based on the requirements of the project (e.g., handling multiple segmentation tasks, etc.).
            x_seg = list(x_seg.values())[0] if x_seg else None

        else:
            x_seg = None

        # Rescale from [0, 1] to relative attenuation values for simulation

        x = volume_preprocessor.rescale_to_HU(x)
        x = volume_preprocessor.rescale_to_attenuation(x) 

        # Simulate scan

        x_adjoint = None
        x_fbp = None
        y = None

        if steps.get("projection", True):
            y = ray_trafo(x)
            if steps.get("add_noise", True):
                y= ray_trafo.simulate_noise(y)
            
            if steps.get("adjoint", True):
                x_adjoint = ray_trafo.backproject(y)
            if steps.get("fbp", True):
                x_fbp = ray_trafo.fbp_reconstruct(y)

  

        if steps.get("visualization", True):
            visualization_dir = study_dir / "visualization"

            # Clean up existing visualizations in the directory to avoid confusion with new visualizations. This can be customized based on the requirements of the project (e.g., keeping old visualizations, etc.).
            
            if visualization_dir.exists():
                import shutil
                shutil.rmtree(visualization_dir)
            visualization_dir.mkdir(parents=True, exist_ok=True)


            visualize_planes(x=x, meta_data=meta_data, vmin=0.0, vmax=2.0, save_dir=visualization_dir, prefix="volume")

            if x_seg is not None:
                visualize_planes(x=x_seg, meta_data=meta_data, vmin=0.0, vmax=4.0, save_dir=visualization_dir, prefix="segmentation")
            if x_adjoint is not None:
                visualize_planes(x=x_adjoint, meta_data=meta_data, vmin=0.0, vmax=2, save_dir=visualization_dir, prefix="adjoint_reconstruction")
            if x_fbp is not None:
                visualize_planes(x=x_fbp, meta_data=meta_data, vmin=0.0, vmax=2, save_dir=visualization_dir, prefix="fbp_reconstruction")
            if y is not None:
                visualize_planes(x=y, meta_data=meta_data, vmin=0.0, vmax=400, save_dir=visualization_dir, prefix="sinogram")


        if steps.get("save_results", False):
            # Save results (e.g., segmentations, reconstructions, etc.) to disk. This can be customized based on the requirements of the project (e.g., saving in a specific format, etc.).
            
            print(f"saving processed volume to {study_dir}...")
            torch.save(x.squeeze(), study_dir / "volume_processed.pt")
            if x_seg is not None:
                print(f"saving segmentation to {study_dir}...")
                torch.save(x_seg.squeeze(), study_dir / "segmentation.pt")
            if y is not None:
                print(f"saving sinogram to {study_dir}...")
                torch.save(y.squeeze(), study_dir / "sinogram.pt")
            print("\n")
    

    # Generate and save run config:

    run_config = {
        "steps": steps,
        "data_cfg": data_cfg,
        "preprocess_cfg": preprocess_cfg,
        "ray_trafo_cfg": ray_trafo_cfg,
        "meta_data": meta_data if meta_data is not None else {},
    }
    run_config_path = Path("data/data_pipeline_run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(_to_jsonable(run_config), f, indent=4)
    print(f"Run config saved to {run_config_path}")
        

        
def main():
    steps = {
        "denoising": False,
        "segmentation": False,
        "projection": True,
        "add_noise": True,
        "adjoint": False,
        "fbp": False,
        "visualization": False,
        "save_results": True,
    }

    # mode = "test"  # "train", "val", or "test"

    for mode in ["train", "val", "test"]:
        data_pipeline(steps=steps, mode=mode)


if __name__ == "__main__":
    main()