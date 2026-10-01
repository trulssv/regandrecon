"""
Bookkeeping of runs (data simulation, training, evaluation, ...): which config, code version and environment produced an output.

A run is tracked with the start_run context manager, which writes into a log directory
    <log_dir>/runs.jsonl                one RunRecord per line, appended when a run ends (completed, failed or interrupted)
    <log_dir>/logs/<run_id>.log         the run's log output (everything logged through the logging module during the run)
    <log_dir>/metrics/<run_id>.jsonl    optional per-step metrics, see RunRecord.log_metrics

The config is everything that determines the output (e.g. preprocessing and ray transform parameters, or model and optimizer
hyperparameters), while args are settings that only control how the run is executed (e.g. which studies, overwrite, number of workers).
pin_config ties an output directory to a single config, so outputs of different configs are never mixed.
"""
import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

from data.utils import _to_jsonable

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_PACKAGES = ("torch", "odl", "astra-toolbox", "numpy")

logger = logging.getLogger(__name__)


class ConfigMismatchError(ValueError):
    """Raised when an output directory was produced with a different config."""


def config_hash(config: dict) -> str:
    """Short, stable hash of a config. Key order does not matter."""
    canonical = json.dumps(_to_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def file_hash(path: Path) -> str:
    """Short hash of a file's content, e.g. to record which version of an input file was used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict:
    """The commit and branch of the repository. dirty is True if there are uncommitted changes, i.e. the commit alone does not reproduce the code."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def environment_info() -> dict:
    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "gpu": gpu,
        "packages": packages,
    }


@dataclass
class RunRecord:
    run_id: str
    kind: str
    config: dict
    config_hash: str
    args: dict
    git: dict
    environment: dict
    started_at: str
    log_dir: str
    finished_at: str | None = None
    duration_s: float | None = None
    status: str = "running"
    results: dict = field(default_factory=dict)  # Summary of the run's output, filled in by the caller.

    def log_metrics(self, step: int, **metrics: float) -> None:
        """Appends one line of metrics (e.g. losses of a training step) to <log_dir>/metrics/<run_id>.jsonl."""
        path = Path(self.log_dir) / "metrics" / f"{self.run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(_to_jsonable({"step": step, "time": time.time(), **metrics})) + "\n")


@contextmanager
def start_run(kind: str, config: dict, log_dir: Path, args: dict | None = None) -> Iterator[RunRecord]:
    """
    Tracks a run: records config, code version and environment, captures the log output in <log_dir>/logs/<run_id>.log
    and appends the RunRecord to <log_dir>/runs.jsonl when the run ends, also if it fails.

    input:
        kind: The kind of run, e.g. "simulation" or "training".
        config: Everything that determines the output of the run.
        log_dir: Directory for the run logs, typically the output directory of the run.
        args: Settings that only control how the run is executed.
    output:
        The RunRecord, whose results the caller can fill in during the run.
    """
    started = datetime.now()
    record = RunRecord(
        run_id=f"{started:%Y%m%d-%H%M%S}_{kind}",
        kind=kind,
        config=_to_jsonable(config),
        config_hash=config_hash(config),
        args=_to_jsonable(args or {}),
        git=git_info(),
        environment=environment_info(),
        started_at=started.isoformat(timespec="seconds"),
        log_dir=str(log_dir),
    )

    log_path = Path(log_dir) / "logs" / f"{record.run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(min(previous_level, logging.INFO) if previous_level else logging.INFO)
    root.addHandler(handler)

    logger.info("Started %s run %s (config %s, commit %s%s)", kind, record.run_id, record.config_hash,
                record.git["commit"], " with uncommitted changes" if record.git["dirty"] else "")
    logger.info("Config: %s", json.dumps(record.config))
    logger.info("Args: %s", json.dumps(record.args))

    try:
        yield record
        record.status = "completed"
    except KeyboardInterrupt:
        record.status = "interrupted"
        raise
    except BaseException:
        record.status = "failed"
        logger.exception("Run %s failed", record.run_id)
        raise
    finally:
        finished = datetime.now()
        record.finished_at = finished.isoformat(timespec="seconds")
        record.duration_s = round((finished - started).total_seconds(), 1)
        logger.info("Finished run %s with status %s after %.1f s. Results: %s", record.run_id, record.status, record.duration_s, json.dumps(_to_jsonable(record.results)))

        with open(Path(log_dir) / "runs.jsonl", "a") as f:
            f.write(json.dumps(_to_jsonable(asdict(record))) + "\n")

        root.removeHandler(handler)
        root.setLevel(previous_level)
        handler.close()


def pin_config(path: Path, config: dict) -> str:
    """
    Ties an output directory to a single config: writes {"config_hash", "config"} to path if it does not exist,
    and raises ConfigMismatchError if it exists with a different config. Returns the config hash.
    """
    h = config_hash(config)
    if path.exists():
        with open(path, "r") as f:
            pinned = json.load(f)
        if pinned["config_hash"] != h:
            raise ConfigMismatchError(
                f"{path.parent} was produced with config {pinned['config_hash']}, but the current config is {h}. "
                f"Use another output directory, or delete {path.parent} to start over. The pinned config is in {path}."
            )
        return h

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"config_hash": h, "config": _to_jsonable(config)}, f, indent=4)
    return h
