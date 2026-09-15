
# This script contains two main functions: assess_data_quality and prepare_datasets. The assess_data_quality function iterates over the dataset, visualizes the data for the user, and allows them to classify the data quality as high, low, or medium. 
# The classifications and metadata are saved in a JSON file. The prepare_datasets function loads the quality assessment results and organizes the data into separate folders based on quality, performing a train-val-test split for the high and medium quality data.



import os
import sys
from pathlib import Path
import argparse
from typing import Any

import torch
import json
from torch.utils.data import random_split
import tqdm


from data.utils import _to_jsonable
from data.data_loaders import get_raw_dataloader, QualityDataset
from visualization.dynamic_visualization import DynamicVisualization


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))





def _save_quality_assessment_atomic(path: Path, payload: dict) -> None:
    """Write JSON atomically to avoid corrupt files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, path)

def assess_data_quality(data_cfg: dict, use_gui_input: bool = False):
    """
    This function iterates over the dataset and lets the user classify the data quality of each study as good, bad or uncertain by visualizing the data. The user input is validated to ensure that only valid classifications are accepted.
       The classification, and the respective file paths, are saved in a json file for later use.
    """

    from matplotlib import pyplot as plt

    data_loader = get_raw_dataloader(data_cfg, batch_size=1, shuffle=False, num_workers=0)

    quality_assessment = {}
    output_path = Path("data/quality_assessment.json")

    for i, batch in enumerate(data_loader):
        print(f"Batch {i}:")

        # Data pipline

        x = batch["data"]
        meta_data = batch["meta_data"]

        study_dir = batch["study_dir"][0] 


        # Visualize the data for the user to assess the data quality

        vis = DynamicVisualization(x, meta_data, vmin=-1, vmax=0)

        if use_gui_input:
            # New optional GUI mode: show animation + classification buttons in same window.
            quality = vis.classify_plane_dynamic(plane="axial", slice_idx=None, prefix="")
            while quality not in ["h", "l", "m"]:
                print("No valid GUI selection was made. Falling back to terminal input.")
                quality = input("Please classify the data quality of this study as high, low or medium (h/l/m): ")
        else:
            # Default behavior preserved.
            vis.visualize_plane_dynamic(plane="axial", slice_idx=None, prefix="")
            quality = input("Please classify the data quality of this study as high, low or medium (h/l/m): ")
            plt.close()  # Close the visualization after the user has made their assessment

            while quality not in ["h", "l", "m"]:
                quality = input("Invalid input. Please classify the data quality of this study as high, low or medium (h/l/m): ")
        quality_assessment[study_dir] = {
            "quality": {"h": "high", "l": "low", "m": "medium"}[quality],
            "meta_data": _to_jsonable(meta_data),
        }

        # Checkpoint after each study to prevent losing progress on later failure.
        _save_quality_assessment_atomic(output_path, quality_assessment)

    # Final write (redundant but explicit)
    _save_quality_assessment_atomic(output_path, quality_assessment)




def prepare_datasets(quality_assessment: dict=None, data_cfg: dict = None, data_dir: Path = Path("/mnt/data/LDDMM")):
    """
    This loads the quality assesment json file and then saves the data accordingly in separate folder for high, low and medium data. Additionally, for the high and medium data, we perform a train-val-test split and save the respective splits in separate folders. The specific steps can be customized based on the requirements of the project.
    """
    if quality_assessment is None:
        with open("data/quality_assessment.json", "r") as f:
            quality_assessment = json.load(f)
    
    if data_cfg is None:
        with open("data/data.json", "r") as f:
            data_cfg = json.load(f)

    
    # Create directories for high, low and medium data
    for quality in ["high", "low", "medium"]:
        (data_dir / quality).mkdir(parents=True, exist_ok=True)

        if quality in ["high", "medium"]:
            # Create train, val, test subdirectories for high and medium data
            for split in ["train", "val", "test"]:
                (data_dir / quality / split).mkdir(parents=True, exist_ok=True)
    

    # Create quality-based data loaders

    quality_datasets = {quality: QualityDataset(quality_assessment, quality=quality) for quality in ["high", "low", "medium"]}


    # For the high and medium data, we perform a train-val-test split and save the respective splits in separate folders. The specific steps can be customized based on the requirements of the project (e.g., random split, patient-wise split, etc.).

    split_ratios = data_cfg.get("train_val_test_split", {"train": 0.7, "val": 0.15, "test": 0.15})
    g = torch.Generator().manual_seed(split_ratios.get("random_seed", 42))  

    for quality in ["high", "medium"]:

        print("Processing quality:", quality)
        input(f"Press Enter to start processing {quality} data...")

        dataset = quality_datasets[quality]
        total_size = len(dataset)
        train_size = int(split_ratios["train"] * total_size)
        val_size = int(split_ratios["val"] * total_size)
        test_size = total_size - train_size - val_size

        

        test_set, train_set, val_set = random_split(dataset, [test_size, train_size, val_size], generator=g)

        for split, split_set in zip(["train", "val", "test"], [train_set, val_set, test_set]):
            split_dir = data_dir / quality / split
            print(f"processing {quality} data for {split} split with {len(split_set)} samples. Saving to {split_dir}...")
            for idx in tqdm.tqdm(split_set.indices):
                study_path = Path(dataset.studies[idx][0])  # Get the study directory for this sample
                meta_data = dataset.studies[idx][1]  # Get the meta data for this sample

                # Create a study spefic id and directory using the patient_id and study_data
                patient_id = meta_data["patient_id"]
                study_date = meta_data["study_date"]
                study_id = f"{patient_id[0]}_{study_date[0]}"
                study_split_dir = split_dir / study_id
                study_split_dir.mkdir(parents=True, exist_ok=True)

                # Save the data and meta data in the specified directory

                vol = torch.load(study_path / "volume.pt")
                torch.save(vol, study_split_dir / "volume.pt")


                meta_data_path = study_split_dir / "meta_data.json"
                with open(meta_data_path, "w") as f:
                    json.dump(meta_data, f)

def main():
    parser = argparse.ArgumentParser(description="Run data pipeline.")
    parser.add_argument(
        "--gui-input",
        action="store_true",
        help="Enable GUI button-based data quality classification.",
    )
    args = parser.parse_args()

    # Default remains terminal input behavior unless --gui-input is provided.

    use_gui_input = args.gui_input
    
    
    with open("data/data.json", "r") as f:
        data_cfg = json.load(f)

    if use_gui_input:
        print("Data quality mode: GUI buttons enabled (Good/Bad/Uncertain).")
    else:
        print("Data quality mode: terminal input (g/b/u).")

    # Keep current default behavior. Set use_gui_input=True to enable in-window buttons.
    # assess_data_quality(data_cfg, use_gui_input=use_gui_input)

    prepare_datasets(quality_assessment=None, data_cfg=data_cfg, data_dir = Path("/mnt/data/LDDMM"))



    

if __name__ == "__main__":
    main()