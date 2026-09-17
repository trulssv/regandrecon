import sys
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Dataset
from data.utils import DataFiles, _unnest_meta_data



REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class QualityDataset(Dataset):
    def __init__(self, quality_assesment: dict, quality: str)->None:
        self.quality_assesment = quality_assesment
        self.quality = quality

        # Load all studies with the specified quality:

        self.studies = [(study, q["meta_data"]) for study, q in quality_assesment.items() if q["quality"] == quality]

    
    def __len__(self)->int:
        return len(self.studies)

    def __getitem__(self, idx: int)->tuple[str, dict]:
        study = self.studies[idx]
        return study


class RawDataset(Dataset):
    def __init__(self, data_files: DataFiles)->None:
        self.data_files = data_files

    def __len__(self)->int:
        return len(self.data_files)

    def __getitem__(self, idx: int)->dict:

        study = self.data_files[idx]
        data, meta_data = self._load_study(study)

        study_dir = self._get_study_dir(study)

        l = {"data": data, "meta_data": meta_data, "study_dir": str(study_dir)}
        return l

    def _get_study_dir(self, study_files: list[Path])->Path:
        """Finds the common study directory for a list of series files."""

        # TODO this might be broken now. Fix docs for this class it is a mess

        series_file = study_files[0]
        return series_file.parent.parent



    def _verify_meta_data(self, meta_data):
        """This function verifies that the meta data is consistent over all series in a a study."""

        items = meta_data[0].items()

        for md in meta_data:
            if md.items() != items:
                raise ValueError(f"Meta data keys are not consistent over all series in a study. Expected keys: {items}, but got {md.items()}.")
        else:
            return True

    def _load_study(self, study):
        """This function loads all the valid series directories from a given study directory."""
        
        meta_data = []
        data = []

        for series_dir in study:
    
            volume, scan_info = self._load_series(series_dir)
            data.append(volume)
            meta_data.append(scan_info)
        
        # Check if the meta data is consistent over all series in a study. If not, raise an error.

        self._verify_meta_data(meta_data)

        # If the meta data is consistent over all series in a study, we can just take the meta data from the first series as representative for the whole study.

        meta_data = meta_data[0]

        # Stack the data from all series in a study into a single tensor. The resulting tensor will have the shape (T, D, H, W) where T, D, H, W are the dimensions of the volume.

        data = torch.stack(data, dim=0)

        return data, meta_data

    def _load_series(self, series_dir):
        """This function loads the data volume.pt and the meta data scan_info.json from a given series directory."""

        for file in series_dir:
            if file.name == "volume.pt":
                volume = torch.load(file)
            elif file.name == "scan_info.json":
                with open(file, "r") as f:
                    scan_info = json.load(f)
        return volume, scan_info

class RegAndReconDataset(Dataset):
    """  
    This is a generic dataset class for the RegAndRecon project. It can be used to load data for training, validation and testing. The dataset is initialized with a list of qualities (e.g., ["high", "medium"]) and a mode (e.g., "train") 
    to specify which data to load. The dataset will then load the volumes projections and meta data for the specified qualities and mode. The specific implementation of the data loading can be customized based on the requirements of the project (e.g., loading from disk, applying preprocessing, etc.).
    """
    def __init__(self, qualities: list[str], mode: str, data_root: Path = Path("/media/trulssv/LDDMM"), files: list[str] = ["volume.pt", "volume_processed.pt", "sinogram.pt", "segmentation.pt", "meta_data.json"])->None:
        super().__init__()
        self.qualities = qualities
        self.mode = mode
        self.data_root = data_root
        self.files = files
        assert mode in ["train", "val", "test"], f"Invalid mode {mode}. Expected one of ['train', 'val', 'test']."
        assert all(q in ["high", "low", "medium"] for q in qualities), f"Invalid quality in {qualities}. Expected all qualities to be one of ['high', 'low', 'medium']."

        self.data_dirs = [self.data_root / q / mode for q in qualities]  # Assuming we want to load data for the specified qualities and mode. This can be customized based on the requirements of the project.
        self.study_dirs = self._get_study_dirs()  # Get the study directories for the specified qualities and mode. This can be customized based on the requirements of the project.

    def _get_study_dirs(self) -> list[Path]:
        """This function gets the study directories for the specified qualities and mode."""
        study_dirs = []
        for data_dir in self.data_dirs:
            for study_dir in data_dir.iterdir():
                if study_dir.is_dir():
                    study_dirs.append(study_dir)
        return study_dirs

    def _load_study(self, study_dir: Path)->dict:
        """This function loads the data and meta data from a given study directory."""

        data = {}
        missing_files = []
        for file in self.files:
            file_path = study_dir / file
            if file_path.exists():
                if file_path.suffix == ".pt":
                    data[file_path.stem] = torch.load(file_path, map_location="cpu")  # Load the tensor to CPU to avoid GPU memory issues. The tensor can be moved to GPU later in the training loop as needed.

                elif file_path.suffix == ".json":
                    with open(file_path, "r") as f:
                        json_data = json.load(f)
                        data[file_path.stem] = _unnest_meta_data(json_data)
            else:
                print(f"Warning: Expected file {file} not found in {study_dir}. Skipping this file.")
                missing_files.append(file)
                # IMPORTANT: default_collate cannot handle None values.
                # Use collate-safe placeholders while preserving key structure.
                if file_path.suffix == ".pt":
                    data[file_path.stem] = torch.empty(0)
                elif file_path.suffix == ".json":
                    data[file_path.stem] = {}

        data["missing_files"] = missing_files
        return data



    def __len__(self)->int:
        return len(self.study_dirs)
    
    def __getitem__(self, idx: int)->dict:
        study_dir = self.study_dirs[idx]
        data = self._load_study(study_dir)
        data["study_dir"] = str(study_dir)  # Add the study directory to the data dictionary. This can be useful for saving results, etc.

        return data


def get_raw_dataloader(data_cfg: dict, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0):
    """This function returns a DataLoader for the RawDataset."""
    data_files = DataFiles(data_cfg)
    dataset = RawDataset(data_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def main():
    pass

if __name__ == "__main__":
    main()