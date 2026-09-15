
import torch

from data.data_loaders import RegAndReconDataset
from pathlib import Path
from operators.ray_transform import DynamicRayTransform
import json
from visualization.dynamic_visualization import DynamicVisualization
from time import time

# Create dataset 


data_root = Path("/mnt/data/LDDMM")
files = ["volume_processed.pt", "sinogram.pt", "meta_data.json"]
dataset = RegAndReconDataset(qualities=["high"], mode="train", data_root=data_root, files=files)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# Create Ray Trafo

preprocess_cfg = json.load(open("data/preprocess.json", "r"))
ray_trafo_cfg = json.load(open("operators/ray_trafo.json", "r"))

ray_trafo = DynamicRayTransform(preprocess_cfg=preprocess_cfg, ray_trafo_cfg=ray_trafo_cfg)



for batch in data_loader:
    volume = batch["volume_processed"]  # Shape: (B, T, D, H, W)
    meta_data = batch["meta_data"]  # List of dictionaries containing meta data for each sample in the batch
    
    start = time()

    sino = ray_trafo(volume)

    print(f"forward projection time: {time() - start} seconds")
    
    start = time()

    vol = ray_trafo.backproject(sino)

    end = time()
    print(f"reconstruction time: {end - start} seconds")

    #B, T, A, D, H = sino.shape

    #sino = sino.view(B, 1, T * A, D, H) # Reshape to (B, 1, T*A, D, H) for visualization

    vis = DynamicVisualization(meta_data, vmin=0, vmax= vol.max().item(), save_dir="test/plots")

    vis.visualize_slices(vol, plane="axial", time_bin=0, prefix="backprojected_volume")

    vis.visualize_plane_dynamic(vol, plane="axial", prefix="backprojected_volume")

    vis = DynamicVisualization(meta_data, vmin=0, vmax=sino.max().item(), save_dir="test/plots")

    vis.visualize_slices(sino, plane="axial", time_bin=0, prefix="sino")
    vis.visualize_plane_dynamic(sino, plane="axial", prefix="sino")

    assert False

# One FBP recon takes roughly 10 seconds
# One FPB takes roughly 0.2 seconds


# Cone:

# forward projection time: 0.35099339485168457 seconds
# reconstruction time: 0.48052287101745605 seconds

# Parallel:

# forward projection time: 0.347034215927124 seconds
# reconstruction time: 0.3073923587799072 seconds