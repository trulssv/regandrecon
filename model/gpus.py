"""
GPU setup of the training scripts: the CUDA settings and the selection of a (free) GPU for torch and ASTRA.

NOTE: this module does not import odl (unlike model.build), since importing odl creates a CUDA context (about 0.4 GB) on ASTRA's current
GPU, GPU 0 unless a GPU was selected before. A distributed run therefore selects its GPUs before it imports the model, see train/train.py.
"""
import logging
import math
import os
import subprocess

import torch

logger = logging.getLogger(__name__)


def configure_cuda(exact_fp32: bool = False) -> None:
    """Settings for the training scripts, to be called before anything is allocated on the GPU (see model/PERFORMANCE.md):
    - expandable segments in the caching allocator, which avoids most of the fragmentation of the large, differently sized
      activations of the volume and sinogram nets.
    - cuDNN autotuning, which picks the fastest convolution algorithms for the (fixed) volume and sinogram shapes on the first
      step (about 25% faster steps at full resolution). The chosen algorithms may differ between runs.
    - exact_fp32: full fp32 instead of TF32 convolutions (cuDNN's default). TF32's 10 bit mantissa is amplified by the large
      sinogram values over the recurrent updates: after one forward pass at full resolution, f differed by 0.3% from fp32, and by
      1.4% between two TF32 algorithms, so with autotuning TF32 results also depend on the algorithms that were picked. fp32 is
      bitwise independent of the algorithm, but makes a step about 45% slower."""
    if torch.cuda.is_initialized():
        logger.warning("configure_cuda was called after CUDA was initialized, so the allocator settings have no effect.")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True
    if exact_fp32:
        torch.backends.cudnn.allow_tf32 = False


def gpu_usage() -> dict[int, tuple[float, float]]:
    """(memory used in GB, utilization in %) of each GPU visible to torch, by torch index, according to nvidia-smi. The GPUs are matched
    by UUID, as torch and nvidia-smi can number them differently (CUDA_DEVICE_ORDER, CUDA_VISIBLE_DEVICES). Creates no CUDA context."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True).stdout
    by_uuid = {}
    for line in out.strip().splitlines():
        uuid, used_mib, utilization = (column.strip() for column in line.split(","))
        by_uuid[uuid.removeprefix("GPU-")] = (float(used_mib) * 2**20 / 1e9, float(utilization) if utilization.isdigit() else 0.0)
    return {i: by_uuid.get(str(torch.cuda.get_device_properties(i).uuid), (math.inf, math.inf)) for i in range(torch.cuda.device_count())}


def free_gpus(max_used_gb: float = 1.0, max_utilization_pct: float = 10.0) -> list[int]:
    """The torch indices of the GPUs that are currently free: less than max_used_gb of memory in use and at most max_utilization_pct
    utilization (a sample, see gpu_usage), in index order.
    NOTE: 1 GB leaves room for the CUDA context (about 0.5 GB) that the other ranks of a distributed run may already have created on
    their GPUs when this rank looks, so that all ranks see the same free GPUs. The index order makes their choices consistent."""
    return [i for i, (used_gb, utilization) in gpu_usage().items() if used_gb < max_used_gb and utilization <= max_utilization_pct]


def select_gpu(local_rank: int = 0, local_world_size: int = 1, gpus: str = "auto") -> torch.device:
    """The GPU of a (local) rank, also set as torch's current device and as ASTRA's GPU. gpus is "auto" for the local_rank-th of the
    currently free GPUs (see free_gpus), or comma separated torch indices, of which the rank takes the local_rank-th.
    NOTE: odl only selects the ASTRA GPU when the reconstruction space lives on a GPU (see odl/applications/tomo/backends/astra_cuda.py),
    which ours does not, so without this every rank would run its ray transforms on GPU 0. ASTRA and torch use the same CUDA runtime,
    and hence the same indices."""
    import astra

    if gpus == "auto":
        candidates = free_gpus()
        if len(candidates) < local_world_size:
            usage = ", ".join(f"GPU {i}: {used:.1f} GB, {util:.0f}%" for i, (used, util) in gpu_usage().items())
            raise RuntimeError(f"Only {len(candidates)} free GPUs ({candidates}) for {local_world_size} processes ({usage}). "
                               "Choose the GPUs explicitly (--gpus / --device) or wait for some to be free.")
    else:
        candidates = [int(i) for i in gpus.split(",")]
        assert len(candidates) >= local_world_size, f"{local_world_size} processes need at least as many GPUs, but got {candidates}."
    index = candidates[local_rank]
    torch.cuda.set_device(index)
    astra.set_gpu_index(index)
    return torch.device("cuda", index)
