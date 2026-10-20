"""
Checks for the run bookkeeping in tracking/run.py.

Run: pytest test/test_tracking.py
"""
import json
import logging

import pytest

from tracking import ConfigMismatchError, config_hash, pin_config, start_run


def read_runs(log_dir) -> list[dict]:
    return [json.loads(line) for line in (log_dir / "runs.jsonl").read_text().splitlines()]


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash({"b": {"d": 3, "c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_start_run_records_completed_run(tmp_path):
    config = {"lr": 1e-3, "model": {"depth": 4}}
    with start_run("training", config, log_dir=tmp_path, args={"num_workers": 2, "out": tmp_path}) as run:
        logging.getLogger("some.module").info("hello from the run")
        run.results["best_loss"] = 0.5
        run.log_metrics(step=0, loss=1.0)
        run.log_metrics(step=1, loss=0.5)

    (record,) = read_runs(tmp_path)
    assert record["status"] == "completed"
    assert record["kind"] == "training"
    assert record["config"] == config and record["config_hash"] == config_hash(config)
    assert record["args"] == {"num_workers": 2, "out": str(tmp_path)}
    assert record["results"] == {"best_loss": 0.5}
    assert record["git"]["commit"] is not None
    assert "torch" in record["environment"]["packages"]
    assert record["duration_s"] is not None

    assert "hello from the run" in (tmp_path / "logs" / f"{record['run_id']}.log").read_text()
    metrics = (tmp_path / "metrics" / f"{record['run_id']}.jsonl").read_text().splitlines()
    assert [json.loads(m)["loss"] for m in metrics] == [1.0, 0.5]


def test_start_run_records_failed_run(tmp_path):
    with pytest.raises(RuntimeError):
        with start_run("simulation", {"a": 1}, log_dir=tmp_path):
            raise RuntimeError("boom")

    (record,) = read_runs(tmp_path)
    assert record["status"] == "failed"
    assert "boom" in (tmp_path / "logs" / f"{record['run_id']}.log").read_text()


def test_start_run_detaches_its_log_handler(tmp_path):
    handlers = list(logging.getLogger().handlers)
    with start_run("simulation", {"a": 1}, log_dir=tmp_path):
        pass
    assert logging.getLogger().handlers == handlers


def test_pin_config(tmp_path):
    path = tmp_path / "config.json"
    h = pin_config(path, {"a": 1})
    assert json.loads(path.read_text()) == {"config_hash": h, "config": {"a": 1}}
    assert pin_config(path, {"a": 1}) == h  # same config: fine
    with pytest.raises(ConfigMismatchError):
        pin_config(path, {"a": 2})


def test_key_step_times_count_nested_regions_once():
    """A region nested in a key step (e.g. the warps of the registered data term) is part of that key step, not a key step of its own,
    and the recompute of a checkpointed block during backward is not counted as forward time."""
    import time

    import torch

    from tracking.timeline import StepTimeline, region

    timeline = StepTimeline(torch.device("cpu"))
    with timeline.record():
        with region("step/forward"):
            with region("hlpd/deform"):
                time.sleep(0.01)
        with region("step/loss"):
            with region("loss/registered"):
                with region("loss/registered/deform"):
                    time.sleep(0.02)
        with region("step/backward"):
            with region("hlpd/deform"):   # a recompute
                time.sleep(0.05)

    times = timeline.history[-1]
    assert "loss/registered/deform" not in times
    assert 0.02 <= times["loss/registered"] < 0.05
    assert 0.01 <= times["hlpd/deform"] < 0.05


def test_prediction_gif_keeps_every_version(tmp_path):
    """With history_dir, every GIF is also kept under a name with its epoch and step, next to the latest one."""
    import torch

    from tracking.training_plots import plot_prediction_gif

    x = [torch.rand(1, 4, 6, 6) for _ in range(2)]
    v = [torch.zeros(1, 4, 6, 6, 3) for _ in range(2)]
    for epoch, step in ((0, 0), (1, 10)):
        plot_prediction_gif(x, x, v, (8.0, 6.0, 6.0), tmp_path / "prediction.gif", step, x_warped=x, epoch=epoch, history_dir=tmp_path / "gifs")

    assert (tmp_path / "prediction.gif").exists()
    assert sorted(p.name for p in (tmp_path / "gifs").iterdir()) == ["prediction_epoch00000_step000000.gif", "prediction_epoch00001_step000010.gif"]
    assert (tmp_path / "prediction.gif").read_bytes() == (tmp_path / "gifs" / "prediction_epoch00001_step000010.gif").read_bytes()
