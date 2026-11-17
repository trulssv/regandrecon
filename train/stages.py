"""
The training stages of train/train.py: consecutive spans of epochs that each train a subset of the nets of the HLPD blocks, with their own
loss terms and learning rate. E.g. first the reconstruction (dual and primal nets) with the data term, then the motion (motion nets) with
the coupling and regularization terms, and finally all nets with all terms at a lower learning rate:

    "training": {..., "stages": [
        {"name": "recon",  "epochs": 100, "train": ["dual", "primal"],
         "loss": {"coupling": {"include": false}, "regularization": {"include": false}, "temporal": {"include": false}}},
        {"name": "motion", "epochs": 100, "train": ["motion"]},
        {"name": "joint",  "epochs": 300, "lr": 1e-5}
    ]}

Each stage has
    name     unique, names its best checkpoint (best_<name>.pt) and can be started at with --start-stage
    epochs   its number of epochs. The stages run back to back, so the training has their sum of epochs ("epochs" of the training config
             is then optional, and must equal the sum if given).
    train    the nets it trains, of "dual", "primal" and "motion" (all by default), see HLPDModel.set_trainable. The others are frozen.
    lr       the learning rate of its optimizer, by default that of the optimizer config
    loss     its changes of the loss config of the model, merged term by term: {"coupling": {"include": false}} only switches the coupling
             term off and keeps its other parameters. The domain and the Helmholtz parameters of the regularization are shared by all
             stages (the latter are those of the Helmholtz operator, which is built once).
Each stage starts with a new optimizer (over the nets it trains) and scheduler, so their state does not carry over between stages.

Without "stages", the training is a single stage named "train" of "epochs" epochs that trains all nets with the loss config of the model,
as before the stages were introduced. Kept free of model imports, so that it can be used before the model is built (see train/train.py).
"""
from dataclasses import dataclass, field

# NOTE: the nets of HLPDModel.set_trainable (model/hlpd.py).
NET_GROUPS = ("dual", "primal", "motion")
DEFAULT_STAGE = "train"


@dataclass(frozen=True)
class Stage:
    name: str
    epochs: int
    train: tuple[str, ...] = NET_GROUPS
    lr: float | None = None
    loss: dict = field(default_factory=dict)
    start: int = 0                  # its first epoch, set by parse_stages

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", tuple(self.train))
        assert self.epochs >= 1, f"Stage '{self.name}' must have at least one epoch, but got {self.epochs}."
        assert self.train and set(self.train) <= set(NET_GROUPS), f"Stage '{self.name}' must train some of {NET_GROUPS}, but got {list(self.train)}."
        assert len(set(self.train)) == len(self.train), f"Stage '{self.name}' lists a net twice: {list(self.train)}."
        assert self.lr is None or self.lr > 0, f"The learning rate of stage '{self.name}' must be positive, but got {self.lr}."
        assert isinstance(self.loss, dict), f"The loss of stage '{self.name}' must be a dict of loss config sections, but got {self.loss}."
        assert "domain" not in self.loss, f"Stage '{self.name}' cannot change the loss domain, which all stages share."

    @property
    def end(self) -> int:
        """One past its last epoch."""
        return self.start + self.epochs


def parse_stages(train_cfg: dict) -> list[Stage]:
    """The stages of the training config, see the module docstring."""
    if "stages" not in train_cfg:
        return [Stage(DEFAULT_STAGE, train_cfg["epochs"])]
    assert train_cfg["stages"], "The training config has an empty list of stages."
    stages, start = [], 0
    for cfg in train_cfg["stages"]:
        unknown = set(cfg) - {"name", "epochs", "train", "lr", "loss"}
        assert not unknown, f"Unknown keys {sorted(unknown)} in stage {cfg}."
        stages.append(Stage(**cfg, start=start))
        start = stages[-1].end
    names = [s.name for s in stages]
    assert len(set(names)) == len(names), f"The stage names must be unique, but got {names}."
    assert train_cfg.get("epochs", start) == start, \
        f"The training config has {train_cfg['epochs']} epochs, but its stages {start}. Leave out 'epochs' with stages."
    return stages


def stage_index(stages: list[Stage], epoch: int) -> int:
    """The index of the stage of an epoch (the last one for epochs past the end)."""
    return next((i for i, s in enumerate(stages) if epoch < s.end), len(stages) - 1)


def stage_by_name(stages: list[Stage], name: str) -> int:
    names = [s.name for s in stages]
    assert name in names, f"No stage '{name}', the stages are {names}."
    return names.index(name)


def merge_loss_config(base: dict, changes: dict) -> dict:
    """The loss config base (sections of LossParams, see model/hlpd_loss.py) with the sections of changes merged into its own."""
    merged = dict(base)
    for name, section in changes.items():
        assert isinstance(section, dict), f"A stage changes the loss section '{name}' with {section}, but it must be a dict."
        merged[name] = {**base.get(name, {}), **section}
    return merged
