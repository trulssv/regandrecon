"""The training stages of train/stages.py: parsing, the stage of each epoch, and the loss config of each stage."""
from dataclasses import asdict

import pytest

from model.hlpd_loss import LossParams
from train.stages import DEFAULT_STAGE, NET_GROUPS, Stage, merge_loss_config, parse_stages, stage_by_name, stage_index


STAGES = [
    {"name": "recon", "epochs": 3, "train": ["dual", "primal"], "loss": {"coupling": {"include": False}, "regularization": {"include": False}}},
    {"name": "motion", "epochs": 2, "train": ["motion"]},
    {"name": "joint", "epochs": 4, "lr": 1e-5},
]


def test_without_stages_the_training_is_a_single_stage() -> None:
    """The training config of runs before the stages trains all nets with the model's loss for its epochs."""
    stages = parse_stages({"epochs": 7})
    assert stages == [Stage(DEFAULT_STAGE, 7)]
    assert stages[0].train == NET_GROUPS and stages[0].lr is None and stages[0].loss == {}
    assert [stage_index(stages, e) for e in range(7)] == [0] * 7


def test_stages_run_back_to_back() -> None:
    stages = parse_stages({"stages": STAGES})
    assert [(s.name, s.start, s.end) for s in stages] == [("recon", 0, 3), ("motion", 3, 5), ("joint", 5, 9)]
    assert [stage_index(stages, e) for e in range(9)] == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert stages[1].train == ("motion",) and stages[2].train == NET_GROUPS and stages[2].lr == 1e-5
    assert stage_by_name(stages, "motion") == 1
    assert parse_stages({"epochs": 9, "stages": STAGES}) == stages


@pytest.mark.parametrize("cfg, match", [
    ({"epochs": 10, "stages": STAGES}, "epochs"),
    ({"stages": []}, "empty"),
    ({"stages": [STAGES[0], STAGES[0]]}, "unique"),
    ({"stages": [{"name": "a", "epochs": 1, "train": ["sigma"]}]}, "must train"),
    ({"stages": [{"name": "a", "epochs": 0}]}, "at least one epoch"),
    ({"stages": [{"name": "a", "epochs": 1, "lr": 1e-5, "optimizer": "SGD"}]}, "Unknown keys"),
    ({"stages": [{"name": "a", "epochs": 1, "loss": {"domain": "projection"}}]}, "domain"),
])
def test_invalid_stages(cfg: dict, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        parse_stages(cfg)


def test_unknown_stage_name() -> None:
    with pytest.raises(AssertionError, match="No stage"):
        stage_by_name(parse_stages({"stages": STAGES}), "registration")


def test_stage_loss_changes_only_the_given_keys() -> None:
    """A stage's loss sections are merged into those of the model config, which keep their other parameters."""
    base = LossParams.from_dict({"domain": "image", "coupling": {"weight": 5.0, "gaps": 3, "fidelity": "huber"},
                                 "temporal": {"include": True, "weight": 1e-3}})
    stage = parse_stages({"stages": STAGES})[0]
    params = LossParams.from_dict(merge_loss_config(asdict(base), stage.loss))
    assert not params.coupling.include and (params.coupling.weight, params.coupling.gaps, params.coupling.fidelity) == (5.0, 3, "huber")
    assert not params.regularization.include
    assert params.temporal.include and params.temporal.weight == 1e-3
    assert params.data == base.data and params.domain == "image"
    # No changes keep the model's loss.
    assert LossParams.from_dict(merge_loss_config(asdict(base), {})) == base


def test_stage_loss_from_a_legacy_loss_config() -> None:
    """The flat loss config of earlier runs is read first, so that the stages change its sections."""
    base = LossParams.from_dict({"lambda_coupling": 2.0, "lambda_reg": 1e-3})
    params = LossParams.from_dict(merge_loss_config(asdict(base), {"coupling": {"include": False}}))
    assert not params.coupling.include and params.coupling.weight == 2.0 and params.regularization.weight == 1e-3
