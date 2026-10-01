"""
Manual quality assessment (triage) of the raw 4D CT studies, the first step of the data pipeline (see data/README.md).
Each study is visualized and classified as high, medium or low quality by the user. The classifications are saved in
data/quality_assessment.json, overwriting any previous assessment.

Run: data/scripts/triage.sh [--no-gui-input]
"""
from pathlib import Path
import argparse

from data.config import DATA_ROOT, QUALITY_ASSESSMENT_PATH
from data.loaders import get_raw_dataloader
from data.utils import _to_jsonable, save_json_atomic
from visualization.dynamic_visualization import DynamicVisualization


def assess_data_quality(use_gui_input: bool = False, data_root: Path = DATA_ROOT, output_path: Path = QUALITY_ASSESSMENT_PATH) -> None:
    """
    This function iterates over the dataset and lets the user classify the data quality of each study as high, low or medium by visualizing the data. The user input is validated to ensure that only valid classifications are accepted.
    The classifications are saved in a json file keyed by study id, together with the study directory, number of time bins and scan info. Any previous assessment in output_path is overwritten.
    """

    from matplotlib import pyplot as plt

    data_loader = get_raw_dataloader(data_root, batch_size=1, shuffle=False, num_workers=0)

    print(f"Assessing {len(data_loader)} studies in data directory: {data_root}")


    quality_assessment = {}

    for i, series in enumerate(data_loader):
        print(f"Study {i}: {series.study_id}")

        x = [volume.unsqueeze(0) for volume in series.data]  # Add a singleton batch dim; (Dynamic)Visualization expects each time bin as (B, D, H, W).
        scan_info = series.scan_info


        # Visualize the data for the user to assess the data quality

        vis = DynamicVisualization(scan_info, vmin=-1, vmax=0)

        if use_gui_input:
            # Show animation + classification buttons in same window.
            quality = vis.classify_plane_dynamic(x, plane="axial", slice_idx=None, prefix="")
            while quality not in ["h", "l", "m"]:
                print("No valid GUI selection was made. Falling back to terminal input.")
                quality = input("Please classify the data quality of this study as high, low or medium (h/l/m): ")
        else:
            vis.visualize_plane_dynamic(x, plane="axial", slice_idx=None, prefix="")
            quality = input("Please classify the data quality of this study as high, low or medium (h/l/m): ")
            plt.close()  # Close the visualization after the user has made their assessment

            while quality not in ["h", "l", "m"]:
                quality = input("Invalid input. Please classify the data quality of this study as high, low or medium (h/l/m): ")
        quality_assessment[series.study_id] = {
            "quality": {"h": "high", "l": "low", "m": "medium"}[quality],
            "study_dir": str(series.study_dir),
            "n_time_bins": len(series.data),
            "scan_info": _to_jsonable(scan_info),
        }

        # Checkpoint after each study to prevent losing progress on later failure.
        save_json_atomic(output_path, quality_assessment)

    # Final write (redundant but explicit)
    save_json_atomic(output_path, quality_assessment)


def main():
    parser = argparse.ArgumentParser(description="Assess the quality of the raw 4D CT studies.")
    parser.add_argument(
        "--gui-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GUI button-based data quality classification (--no-gui-input for terminal input).",
    )
    args = parser.parse_args()

    print(f"Data quality mode: {'GUI buttons' if args.gui_input else 'terminal input (h/l/m)'}.")
    assess_data_quality(use_gui_input=args.gui_input)



if __name__ == "__main__":
    main()
