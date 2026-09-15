from train.train_parallel import ParallelHLPDTrainer
from pathlib import Path
import json
import torch

def main():

    config_path = "train/train.json"

    with open(config_path, "r") as f:
        train_config = json.load(f)

    trainer = ParallelHLPDTrainer(train_config=train_config)


    # Checkpoint path:

    checkpoint_dir = Path(trainer.train_config.get("training", {}).get("checkpoint_dir", "checkpoints"))
    run_dir = "20260409_170852"
    epoch = 49

    checkpoint_path = checkpoint_dir / run_dir / f"checkpoint_epoch_{epoch}.pt"

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        trainer.eval(checkpoint_path=checkpoint_path)
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Please check the path and try again.")

if __name__ == "__main__":
    main()