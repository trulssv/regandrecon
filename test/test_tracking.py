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
