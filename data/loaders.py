"""
Data loaders for the two ends of the data pipeline (see data/README.md):
- Raw loader: the raw 4D CT studies under DATA_ROOT, used by the triage (data/triage.py) and the pipeline (data/pipeline.py).
- Quality loader: the simulated studies under SIMULATED_ROOT / <geometry>, filtered by quality and split, used for training.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List

import torch
from torch.utils.data import DataLoader, Dataset

from data.config import DATA_ROOT, DEFAULT_SIMULATED_ROOT, SKIP_STUDIES


# ---------------------------------------------------------------------------
# Raw loader
# ---------------------------------------------------------------------------

GATED_PERCENTAGE_RE = re.compile(r"(\d+\.?\d*)%")

@dataclass
class Raw4DCTSeries:
    """Class representing a raw 4D CT series. This stores the metadata and spatiotemporal sequence of 3D scans."""
    data_root: Path
    patient_dir: Path
    study_dir: Path
    series_dirs: List[Path]
    scan_info: dict
    data: List[torch.Tensor] # The spatiotemporal sequence where each of the list corresponds to 3D scan
    scan_time: List[float]   #  The time points corresponding to each 3D scan in the spatiotemporal sequence

    @property
    def study_id(self) -> str:
        return study_id_from_dir(self.study_dir)


def study_id_from_dir(study_dir: Path) -> str:
    """Unique and stable study identifier, e.g. "patient_100_HM10395/study_35220329" -> "100_HM10395_study_35220329"."""
    return f"{study_dir.parent.name.removeprefix('patient_')}_{study_dir.name}"


def _gated_series_dirs(study_dir: Path) -> list[tuple[float, Path]]:
    """Returns the (gated percentage, series_dir) pairs of a study, ordered by gated percentage."""
    gated_series = []
    for series_dir in study_dir.glob("series_*"):
        if not series_dir.is_dir():
            continue

        match = GATED_PERCENTAGE_RE.search(series_dir.name)
        if match is None:
            print(f"Warning: Skipping series {series_dir}, could not parse gated percentage from name.")
            continue

        gated_series.append((float(match.group(1)), series_dir))

    gated_series.sort(key=lambda item: item[0])
    return gated_series


def index_studies(data_root: Path = DATA_ROOT, skip_studies: Iterable[str] = SKIP_STUDIES) -> list[Path]:
    """Lists all study directories under data_root following the "patient_*_*/study_*" layout, without loading any data.
    Studies whose name contains an entry of skip_studies, or that hold no gated series with volume.pt, are left out."""

    study_dirs = []
    for study_dir in sorted(data_root.glob("patient_*_*/study_*")):
        if not study_dir.is_dir() or any(s in study_dir.name for s in skip_studies):
            continue

        if not any((series_dir / "volume.pt").exists() for _, series_dir in _gated_series_dirs(study_dir)):
            print(f"Warning: Skipping {study_dir}, no valid gated series found.")
            continue

        study_dirs.append(study_dir)
    return study_dirs


def load_study(study_dir: Path, data_root: Path = DATA_ROOT) -> Raw4DCTSeries:
    """Load a raw 4D CT study. The study directory holds one series_* directory per respiratory/cardiac gated phase (e.g.
    "..._Gated,_0.0%_5", "..._Gated,_10.0%_6", ...); these are loaded and ordered by that gated percentage to
    form the study's spatiotemporal sequence. Metadata is shared across a study, so scan_info is taken from
    whichever gated phase loads first.

    input:
        study_dir: The study directory, following the "patient_*_*/study_*" layout.
        data_root: The root directory containing the raw 4D CT data.
    output:
        The Raw4DCTSeries of the study.
    """

    data = []
    scan_time = []
    series_dirs = []
    scan_info = None

    for percentage, series_dir in _gated_series_dirs(study_dir):
        volume_path = series_dir / "volume.pt"
        scan_info_path = series_dir / "scan_info.json"

        if not volume_path.exists() or not scan_info_path.exists():
            print(f"Warning: Skipping series {series_dir}, missing volume.pt or scan_info.json.")
            continue

        data.append(torch.load(volume_path, map_location="cpu"))
        scan_time.append(percentage)
        series_dirs.append(series_dir)

        if scan_info is None:
            with open(scan_info_path, "r") as f:
                scan_info = json.load(f)

    if not data or scan_info is None:
        raise FileNotFoundError(f"No valid gated series found in {study_dir}.")

    return Raw4DCTSeries(
        data_root=data_root,
        patient_dir=study_dir.parent,
        study_dir=study_dir,
        series_dirs=series_dirs,
        scan_info=scan_info,
        data=data,
        scan_time=scan_time,
    )


def data_iterator(data_root: Path = DATA_ROOT) -> Iterator[Raw4DCTSeries]:
    """Iterate over all raw 4D CT studies found under data_root, loading one study at a time."""
    for study_dir in index_studies(data_root):
        yield load_study(study_dir, data_root)



class RawDataSet(Dataset):
    """Lazy dataset over the raw 4D CT studies: only the study directories are indexed on construction,
    and each study is loaded from disk in __getitem__."""
    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root
        self.study_dirs = index_studies(data_root)

    def __len__(self):
        return len(self.study_dirs)

    def __getitem__(self, idx):
        return load_study(self.study_dirs[idx], self.data_root)


def get_raw_dataloader(data_root: Path = DATA_ROOT, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """Returns a DataLoader over RawDataSet. Raw4DCTSeries instances hold Paths, dicts and variable-length
    tensors, none of which default_collate can stack, so this loader only ever runs at batch_size=1 and each
    batch is unwrapped back to a single Raw4DCTSeries."""

    dataset = RawDataSet(data_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )


# ---------------------------------------------------------------------------
# Quality loader
# ---------------------------------------------------------------------------

class RegAndReconDataset(Dataset):
    """
    Quality loader over the output of the data pipeline (data/pipeline.py), organized as <data_root>/<quality>/<mode>/<study_id>,
    where data_root is SIMULATED_ROOT / <geometry> (data/config.py).
    Each item is a dict with
        "sinogram": (T, views, cols, rows) projections, noisy if a noise model is given,
        "meta": the study's meta data (scan info, extent, time steps, ...),
        "study_id": the study id,
        "volume": (T, D, H, W) attenuation values in mm^-1, only if load_volume is set.
    """
    def __init__(self, qualities: list[str], mode: str, data_root: Path = DEFAULT_SIMULATED_ROOT, load_volume: bool = False, noise: Callable[[torch.Tensor], torch.Tensor] | None = None)->None:
        super().__init__()
        assert mode in ["train", "val", "test"], f"Invalid mode {mode}. Expected one of ['train', 'val', 'test']."
        assert all(q in ["high", "low", "medium"] for q in qualities), f"Invalid quality in {qualities}. Expected all qualities to be one of ['high', 'low', 'medium']."

        self.qualities = qualities
        self.mode = mode
        self.data_root = data_root
        self.load_volume = load_volume
        self.noise = noise

        # Only studies that the data pipeline has finished are included.
        self.study_dirs = sorted(d for q in qualities for d in (data_root / q / mode).glob("*/") if (d / "done").exists())

    def __len__(self)->int:
        return len(self.study_dirs)

    def __getitem__(self, idx: int)->dict:
        study_dir = self.study_dirs[idx]

        with open(study_dir / "meta.json", "r") as f:
            meta = json.load(f)

        sinogram = torch.load(study_dir / "sinogram.pt", map_location="cpu")
        if self.noise is not None:
            sinogram = self.noise(sinogram)

        item = {"sinogram": sinogram, "meta": meta, "study_id": meta["study_id"]}
        if self.load_volume:
            item["volume"] = torch.load(study_dir / "volume.pt", map_location="cpu")
        return item


def collate_time_list(batch: list[dict]) -> dict:
    """Collates RegAndReconDataset items into the model-side format: tensors become a list over time of (B, ...) tensors,
    while meta and study_id stay lists of length B. Requires all studies in the batch to have the same number of time bins.

    NOTE: Each study carries its own physical extent (meta["extent"]). All studies currently share the same extent, so a single ray transform
    serves the whole batch. TODO: check or enforce equal extents within a batch if studies with other extents are added."""
    collated = {"meta": [item["meta"] for item in batch], "study_id": [item["study_id"] for item in batch]}
    for key in ("sinogram", "volume"):
        if key in batch[0]:
            collated[key] = list(torch.stack([item[key] for item in batch]).unbind(1))  # (B, T, ...) -> T x (B, ...)
    return collated


def get_quality_loader(
        qualities: list[str],
        mode: str,
        batch_size: int = 1,
        shuffle: bool | None = None,
        num_workers: int = 0,
        load_volume: bool = False,
        noise: Callable[[torch.Tensor], torch.Tensor] | None = None,
        data_root: Path = DEFAULT_SIMULATED_ROOT,
        ) -> DataLoader:
    """Returns a DataLoader over RegAndReconDataset that yields batches in the format of collate_time_list. Shuffles by default only in training mode."""
    dataset = RegAndReconDataset(qualities=qualities, mode=mode, data_root=data_root, load_volume=load_volume, noise=noise)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        collate_fn=collate_time_list,
    )
