import json
import os
from pathlib import Path
from typing import Any

import torch


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically to avoid corrupt files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=4)
    os.replace(tmp_path, path)


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
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    # numpy scalar support without importing numpy explicitly
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def study_extent(scan_info: dict, shape: tuple[int, ...]) -> tuple[float, float, float]:
    """Physical (D, H, W) extent in mm of a raw volume with the given shape. The extent is unchanged by resizing the volume."""
    d, h, w = shape
    pixel_spacing = scan_info["resampled_pixel_spacing"]  # (W spacing, H spacing), mm
    slice_thickness = scan_info["resampled_slice_thickness"]  # mm
    return (d * slice_thickness, h * pixel_spacing[1], w * pixel_spacing[0])