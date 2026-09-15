import torch
from pathlib import Path
from typing import Any

class DataFiles:
    """This class is responsible for handling all the file paths in the specified directory following the pre-defined dataset ordering. 
    It provides methods to search for valid study directories, series directories, and data files, while also allowing for the exclusion of specific studies based on a skip list."""
    def __init__(self, data_cfg):

        self.DATA_DIR = data_cfg["ROOT"]
        self.dataset_ordering = data_cfg["ORGANIZATION"]
        self.skip_studies = set(data_cfg["SKIP_STUDIES"])



        self.study_dirs = self._get_study_directories()
        self.data_paths = self._get_data_paths()
        self.files = self.get_files()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.files[idx]

    def _get_study_directories(self):
        """This function searches for all valid study directories in the specified directory following the pre-defined dataset ordering."""
        data_dir = Path(self.DATA_DIR)
        study_dirs = list(data_dir.glob(f"{self.dataset_ordering[0]}**/{self.dataset_ordering[1]}**/"))

        # Filter out study directories that are in the skip list
        study_dirs = [d for d in study_dirs if  not any(s in d.name for s in self.skip_studies)]


        if len(study_dirs) == 0:
            raise FileNotFoundError(f"No study directories found in {self.DATA_DIR} following the ordering {self.dataset_ordering}.")
        return study_dirs


    def _get_study(self, study_dir):
        """This function loads all the valid series directories from a given study directory."""
        
        series_dirs = sorted(list(study_dir.glob(f"{self.dataset_ordering[2]}**/")))
        if len(series_dirs) == 0:
            print(f"No series directories found in study directory {study_dir} following the ordering {self.dataset_ordering}.")
            return []
        return series_dirs





    
    def _get_series(self, series_dir):
        """This function loads the data from a given series directory."""
        # Implement the logic to load data from the series directory

        series_files = list(series_dir.glob(f"*"))
        series_files = [f for f in series_files if f.is_file()]
        valid_names = ["volume.pt", "scan_info.json"] 
        series_files = [f for f in series_files if f.name in valid_names]
        
        return series_files

    def get_files(self):
        """This function returns a list of all valid data files in the specified directory."""
        all_files = []
        for study_dir in self.study_dirs:
            study_files = [] 
            dirs = self._get_study(study_dir)
            for series_dir in dirs:
                series_files = self._get_series(series_dir)
                study_files.append(series_files)
            all_files.append(study_files)
        return all_files


    def _get_data_paths(self):

        """This function searches and return all files in the specified directory following the pre-defined dataset ordering.
        
        returns:
            A nested list of file paths, where the first level corresponds to patients, the second level corresponds to studies, and the third level corresponds to series. At the series level, the list contains the data volume.pt and metadata scan_info.json.
        """

        data_dir = Path(self.DATA_DIR)
        all_paths = list(data_dir.glob(f"{self.dataset_ordering[0]}**/{self.dataset_ordering[1]}**/{self.dataset_ordering[2]}"))
        return all_paths



def _unnest_meta_data(meta_data: dict)->dict:
    """This function only performs unnesting to depth 2, which is sufficient for the current meta data format. If the meta data format changes in the future, this function might need to be updated to handle deeper nesting."""
    for key, value in meta_data.items():
        if isinstance(value, list) and len(value) == 1:
            meta_data[key] = value[0]
        elif isinstance(value, list) and len(value) > 1:
            for i, v in enumerate(value):
                if isinstance(v, list) and len(v) == 1:
                    value[i] = v[0]
    return meta_data

def _to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors and numpy scalars/arrays to JSON-serializable Python objects."""
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # numpy scalar support without importing numpy explicitly
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return obj





def main():
    
    import json
    with open("./data/data.json", "r") as f:
        data_cfg = json.load(f)
    
    data_files = DataFiles(data_cfg)
    print(len(data_files.files))
    print(len(data_files.files[0]))
    print(len(data_files.files[0][0]))




if __name__ == "__main__":
    main()