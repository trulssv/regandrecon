from data.data_loaders import RegAndReconDataset
from pathlib import Path
import torch

# Create dataset 


data_root = Path("/mnt/data/LDDMM")
files = ["volume_processed.pt", "sinogram.pt", "meta_data.json"]
dataset = RegAndReconDataset(qualities=["high"], mode="train", data_root=data_root, files=files)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

for batch in data_loader:
    volume = batch["volume_processed"]  # Shape: (B, T, D, H, W)
    sinogram = batch["sinogram"]  # Shape: (B, T, num_projections, detector_size)
    meta_data = batch["meta_data"]  # List of dictionaries containing meta data for each sample in the batch

    # Print histogram of the sinogram to check for outliers
    print(f"Sinogram histogram: min={sinogram.min().item()}, max={sinogram.max().item()}, mean={sinogram.mean().item()}, std={sinogram.std().item()}")

    # Print number of entrires close to the maximum value to check for saturation
    num_saturated_entries = (sinogram >= sinogram.max() * 0.99).sum().item()
    print(f"Number of saturated entries in the sinogram: {num_saturated_entries / sinogram.numel() * 100:.2f}%")

    # Remove outliers by clipping the sinogram values to a reasonable range based on the histogram analysis
    sinogram_clipped = torch.clamp(sinogram, min=0, max=0.6 * sinogram.max())

    # Print the histogram of the clipped sinogram to verify that the outliers have been removed
    print(f"Clipped sinogram histogram: min={sinogram_clipped.min().item()}, max={sinogram_clipped.max().item()}, mean={sinogram_clipped.mean().item()}, std={sinogram_clipped.std().item()}")  



    input("Press Enter to continue to the next batch...")

