import torch

# Distributed training imports

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Progress bar and utilities

from tqdm import tqdm
from pathlib import Path
import json
import os
from datetime import datetime

# Model stuff

from data.data_loaders import RegAndReconDataset
from model.model import HLPDModel, HLPDModelWithLoss, ResidualHLPDModelWithLoss


# Visualization

from visualization.dynamic_visualization import DynamicVisualization
from train.train import HLPDTrainer

def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        print("Distributed training is not set up. Running in single GPU mode.")
        return False, 0, 1, 0 # distributed, rank, world_size, local_rank
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return True, rank, world_size, local_rank

def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

def reduce_loss_dict(loss_dict: dict[str, torch.Tensor], world_size: int)->dict[str, torch.Tensor]:
    """Reduce the loss dictionary from all processes so that process with rank 0 has the averaged results. Returns a dict with the same fields as loss_dict, after reduction."""
    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        values = []
        for key in sorted(loss_dict.keys()):
            keys.append(key)
            values.append(loss_dict[key])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
        reduced_loss_dict = {key: value for key, value in zip(keys, values)}
    return reduced_loss_dict


class ParallelHLPDTrainer(HLPDTrainer):

    def __init__(
            self, 
            train_config: dict = None,
            qualities: list[str] = ["high"], 
            verbose: bool = True
        )->None:

        # Setup training with DDP

        self.distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        if not self.distributed:
            print("Running in single GPU mode. For distributed training, please set up the environment variables RANK, WORLD_SIZE, and LOCAL_RANK according to the requirements of your distributed training setup (e.g., using torchrun or a job scheduler).")
        else:
            print(f"Running in distributed mode with rank {self.rank} out of {self.world_size} processes. Local rank is {self.local_rank}.")

        self.is_main = (self.rank == 0) 

        # NOTE: We need to local batch size, and require that the global batch size is divisible by the world size. This is a common requirement for distributed training, as it ensures that each process gets an equal portion of the data and can effectively utilize the available GPUs. If this requirement is not met, we will raise an assertion error to alert the user to adjust the batch size or the number of processes accordingly.

        if self.distributed:
            global_bs = train_config["batch_size"]
            assert global_bs % self.world_size == 0, f"Global batch size {global_bs} must be divisible by world size {self.world_size} for distributed training."
            local_bs = global_bs // self.world_size
            train_config["batch_size"] = local_bs  # Update the batch size in the training configuration to the local batch size for each process.
            print(f"Global batch size {global_bs} is divisible by world size {self.world_size}. Using local batch size {local_bs} for each process.")


        # important: per-process device
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")


        super().__init__(train_config=train_config, qualities=qualities, verbose=verbose, device=device)

    def _init_data_loader(self, qualities: list[str], mode: str)->RegAndReconDataset:

        # NOTE: we only consider the processed volume, the simulated CT data and the meta data
        files = ["volume_processed.pt", "sinogram.pt", "meta_data.json"]
        dataset = RegAndReconDataset(qualities=qualities, mode=mode, data_root=self.data_root, files=files)

        sampler = None
        shuffle = (mode == "train")  # Shuffle only for training data
        if self.distributed:
            sampler = DistributedSampler(
                dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=shuffle,
                drop_last=True  # Drop the last incomplete batch if the dataset size is not divisible by the number of processes. This can help ensure that all batches have the same size, which can be important for certain training setups (e.g., when using BatchNorm). Set to False if you want to keep all samples, even if it results in smaller batches for the last few iterations.
                )
            
            shuffle = False  # Shuffle is handled by the DistributedSampler
        


        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            sampler=sampler, 
            num_workers=4, 
            pin_memory=True
        )
        return data_loader


    def _init_model_and_loss(self)->HLPDModel:
        model_config_path = Path("model/model.json")
        assert model_config_path.exists(), FileNotFoundError(f"Model config file not found at {model_config_path}. Please make sure the model config file is present and has the correct format.")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        lambda_channels = model_config.get("lambda_channels", None)
        gamma_channels = model_config.get("gamma_channels", None)
        sigma_channels = model_config.get("sigma_channels", None)
        hlpd_iterations = model_config.get("hlpd_iterations", 10)
        dropout = model_config.get("dropout", 0.0)
        batch_norm = model_config.get("batch_norm", True)

        model = ResidualHLPDModelWithLoss(
            lambda_channels=lambda_channels, 
            gamma_channels=gamma_channels, 
            sigma_channels=sigma_channels, 
            hlpd_iterations=hlpd_iterations, 
            dropout=dropout, batch_norm=batch_norm, 
            device=self.device, batch_size=self.batch_size
            ).to(self.device)
        
        if self.distributed:
            model = DDP(
                model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False # Set to True if your model has unused parameters. This can help avoid errors related to unused parameters during the backward pass, but it may introduce some overhead. Set it to False if you are sure that all parameters are used in the forward pass.
            )
        return model, None
    
    def _setup_checkpointing(self, checkpoint_dir: Path)->Path:
        """Creates a unique directory for saving checkpoints based on the current date and time. This can be useful for organizing checkpoints from different training runs and avoiding overwriting existing checkpoints. The directory is created under the specified checkpoint directory, and the name of the subdirectory is based on the current date and time in the format "YYYYMMDD_HHMMSS". The function also ensures that the checkpoint directory exists by creating it if it does not already exist."""
        if self.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_checkpoint_dir = checkpoint_dir / time_stamp
            run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            return run_checkpoint_dir

    def _save_checkpoint(self, epoch: int, model: HLPDModel, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, losses: dict[str, list[float]], QM: list[dict[str, float]])->None:
        """This function can be used to save the model checkpoint at regular intervals during training. The checkpoint can include the model state, optimizer state, scheduler state, and any other relevant information such as the current epoch and the training and validation losses. This can be useful for resuming training from a specific checkpoint in case of interruptions, or for analyzing the training process after it has completed."""
        if not self.is_main:
            return  # Only the main process should save checkpoints to avoid conflicts and redundant saves in distributed training.
        
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()


        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "losses": losses,
            "quality_measures": QM
        }
        checkpoint_path = self.run_checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


    def train(self)->None:

        # Set up checkpointing directory for this training run

        self.run_checkpoint_dir = self._setup_checkpointing(self.checkpoint_dir)

        losses = {"train": [], "val": []} 
        QM = []  # Initialize a list to store quality measures for each epoch

        train_sampler = self.data_loaders[0].sampler
        validation_sampler = self.data_loaders[1].sampler

        for epoch in tqdm(range(self.num_epochs), desc="Training Progress", disable=not self.is_main):
            if self.distributed and isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)  # Set the epoch for the DistributedSampler to ensure proper shuffling of the data across epochs in distributed training.
            if self.distributed and isinstance(validation_sampler, DistributedSampler):
                validation_sampler.set_epoch(epoch)  # Set the epoch for the DistributedSampler to ensure proper shuffling of the data across epochs in distributed training.

            train_loss, out_train = self._train_one_epoch(epoch)
            val_loss, out_val = self._validate(epoch)
            self.scheduler.step()

            if self.is_main:
                

                # Save losses for logging

                losses["train"].append(train_loss)
                losses["val"].append(val_loss)

                if (epoch + 1) % self.log_interval == 0:
                    self._plot_losses(losses)



                    self._plot_eval(
                        x_pred=out_val["x"] if out_val is not None else None,
                        f_pred=out_val["f"] if out_val is not None else None,
                        x_gt=out_val["x_gt"] if out_val is not None else None,
                        y=out_val["y"] if out_val is not None else None,
                        v=out_val["v"] if out_val is not None else None,
                        meta_data=out_val["meta_data"] if out_val is not None else None,
                        epoch=epoch+1,
                        mode="val"
                    )

                    self._plot_eval(
                        x_pred=out_train["x"] if out_train is not None else None,
                        f_pred=out_train["f"] if out_train is not None else None,
                        x_gt=out_train["x_gt"] if out_train is not None else None,
                        y=out_train["y"] if out_train is not None else None,
                        v=out_train["v"] if out_train is not None else None,
                        meta_data=out_train["meta_data"] if out_train is not None else None,
                        epoch=epoch+1,
                        mode="train"
                    )

                    QM_epoch = self._get_quality_measures(
                        x_pred=out_train["x"] if out_train is not None else None,
                        x_gt=out_train["x_gt"] if out_train is not None else None,
                        v=out_train["v"] if out_train is not None else None
                    )
                    QM.append(QM_epoch)

                if (epoch + 1) % self.save_interval == 0:
                    self._save_checkpoint(epoch, self.model, self.optimizer, self.scheduler, losses, QM)
        


    def _train_one_epoch(self, epoch: int)->dict[str, float]:
        self.model.train()
        train_loader = self.data_loaders[0]  # Assuming the first data loader is for training. This can be customized based on the requirements of the project (e.g., handling multiple data loaders, etc.).
        
        
        epoch_loss = {}   

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", disable=not self.is_main, leave=True)

        final, out = None, None

        for batch in pbar:
            # Implement the training logic for one epoch, which typically involves iterating over the batches of data, computing the loss, and updating the model parameters using the optimizer.
            

            x_gt = batch["volume_processed"].to(self.device)  # Ground truth volume
            y = batch["sinogram"].to(self.device)  # Measured sinogram

            out, loss = self.model(y, x_gt)  # Predicted volume from the model


            reduced_loss = reduce_loss_dict(loss, self.world_size) # NOTE: we need to reduce the loss across all processes to get the correct loss values for logging and visualization. This is important because each process will have its own local loss values based on its portion of the data, and we want to aggregate these values to get a global view of the training progress.
            
            # Backpropagation and optimization step
                        
            total_loss = loss["total_loss"]
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.is_main:
                
                # Log losses and add to epoch_loss for visualization
                
                for key, value in reduced_loss.items():
                    if key not in epoch_loss:
                        epoch_loss[key] = []
                    epoch_loss[key].append(value.item())

                pbar.set_postfix(
                   {key: f"{value.item():.4f}" for key, value in reduced_loss.items()}    
             )

        # Compute average loss for the epoch
        epoch_loss = {key: sum(values) / len(values) for key, values in epoch_loss.items()}

        if self.is_main:

            final = {key: out.cpu().float() for key, out in out.items()} if out is not None else None  # Move the final output to CPU for visualization and saving results. This can be customized based on the requirements of the project (e.g., handling multiple outputs, etc.).
            final["x_gt"] = x_gt.cpu().float()  # Add the ground truth volume to the final output for visualization and saving results. This can be useful for comparing the predicted volume with the ground truth during evaluation, etc. 
            final["y"] = y.cpu().float()  # Add the measured sinogram to the final output for visualization and saving results. This can be useful for analyzing the results in the context of the specific measurements being processed, etc. 
            final["meta_data"] = batch["meta_data"]  # Add the meta data to the final output for visualization and saving results. This can be useful for analyzing the results in the context of the specific study or sample being processed, etc.

        return epoch_loss, final

    def _validate(self, epoch: int)->None:
        self.model.eval()
        val_loader = self.data_loaders[1]  # Assuming the second data loader is for validation. This can be customized based on the requirements of the project (e.g., handling multiple data loaders, etc.).

        epoch_val_loss = {}
        final, out = None, None

        with torch.no_grad():
            

            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", disable=not self.is_main, leave=True)

            for batch in pbar:
                # Implement the validation logic for one epoch, which typically involves iterating over the batches of validation data, computing the loss, and optionally saving the model if it achieves a new best performance on the validation set.
                
                x_gt = batch["volume_processed"].to(self.device)  # Ground truth volume
                y = batch["sinogram"].to(self.device)  # Measured sinogram

                out, loss = self.model(y, x_gt)  # Predicted volume from the model

                reduced_loss = reduce_loss_dict(loss, self.world_size) # NOTE: we need to reduce the loss across all processes to get the correct loss values for logging and visualization. This is important because each process will have its own local loss values based on its portion of the data, and we want to aggregate these values to get a global view of the training progress.
                
                if self.is_main:

                    # Log losses and add to epoch_val_loss for visualization
                    for key, value in reduced_loss.items():
                        if key not in epoch_val_loss:
                            epoch_val_loss[key] = []
                        epoch_val_loss[key].append(value.item())

                    if self.verbose:
                        pbar.set_postfix({
                            key: f"{value.item():.4f}" for key, value in reduced_loss.items()
                        })

        # Compute average validation loss for the epoch

        epoch_val_loss = {key: sum(values) / len(values) for key, values in epoch_val_loss.items() if len(values) > 0}

        if self.is_main:

            final = {key: out.cpu().float() for key, out in out.items()} if out is not None else None  # Move the final output to CPU for visualization and saving results. This can be customized based on the requirements of the project (e.g., handling multiple outputs, etc.).
            final["x_gt"] = x_gt.cpu().float()  # Add the ground truth volume to the final output for visualization and saving results. This can be useful for comparing the predicted volume with the ground truth during evaluation, etc. 
            final["y"] = y.cpu().float()  # Add the measured sinogram to the final output for visualization and saving results. This can be useful for analyzing the results in the context of the specific measurements being processed, etc. 
            final["meta_data"] = batch["meta_data"]  # Add the meta data to the final output for visualization and saving results. This can be useful for analyzing the results in the context of the specific study or sample being processed, etc.

        return epoch_val_loss, final
    
    def eval(self, checkpoint_path: Path)->None:
        """This function can be used to evaluate the model on the test set using a specific checkpoint. The function loads the model state from the checkpoint, sets the model to evaluation mode, and then iterates over the test data to compute the loss and quality measures. The results can be logged and visualized as needed."""
        

        # Update run_checkpoint dir:

        self.run_checkpoint_dir = checkpoint_path.parent  # Set the run checkpoint directory to the parent directory of the checkpoint path. This can be useful for organizing the evaluation results and visualizations in the same directory as the checkpoint being evaluated, etc.

        print(checkpoint_path)

        def load_checkpoint(checkpoint_path: Path) -> tuple[dict, list[dict[str, float]]]:
            """This function loads the model state from the specified checkpoint path. It returns the model state, optimizer state, scheduler state, and any other relevant information that was saved in the checkpoint."""
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please make sure the checkpoint file is present and has the correct format.")
        
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            QM = checkpoint.get("quality_measures", None)


            state_dict = checkpoint.get("model_state_dict", None)
            if state_dict is None:
                raise KeyError(f"Model state dict not found in checkpoint at {checkpoint_path}. Please make sure the checkpoint file contains the model state dict under the key 'model_state_dict'.")
            return state_dict, QM
        
        def plot_QM(QM: list[dict[str, float]])->None:
            """This function can be used to plot the quality measures over epochs. The input QM is a list of dictionaries, where each dictionary contains the quality measures for a specific epoch. The function can create plots for each quality measure and save them to the specified directory."""

            from matplotlib import pyplot as plt            

            save_dir = self.run_checkpoint_dir / "quality_measures"
            save_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(QM, list) and all(isinstance(qm, dict) for qm in QM):
                # Extract quality measure names and values
                qm_names = QM[0].keys() if QM else []
                qm_values = {name: [qm[name] for qm in QM] for name in qm_names}

                # Create plots for each quality measure
                for name in qm_names:
                    plt.figure()
                    plt.plot(qm_values[name], marker='o')
                    plt.title(f"{name} over Epochs")
                    plt.xlabel("Epoch")
                    plt.ylabel(name)
                    plt.grid()
                    plt.savefig(save_dir / f"{name}_over_epochs.png")
                    plt.close()
            else:
                print(f"Warning: QM is not in the expected format (list of dictionaries). Cannot plot quality measures. QM: {QM}")




        model = self.model.module if isinstance(self.model, DDP) else self.model  # Get the underlying model if using DDP, otherwise use the model directly.
        model_state_dict, QM = load_checkpoint(checkpoint_path)

        # Plot quality measures if available in the checkpoint
        if QM is not None:
            plot_QM(QM)

        model.load_state_dict(model_state_dict)
        model.eval()

        final, out = None, None

        test_loader = self.data_loaders[2]  # Assuming the third data loader is for testing. This can be customized based on the requirements of the project (e.g., handling multiple data loaders, etc.).        

        with torch.no_grad():

            pbar = tqdm(test_loader, desc="Evaluating on test set", disable=not self.is_main, leave=True)

            for batch in pbar:
                x_gt = batch["volume_processed"].to(self.device)  # Ground truth volume
                y = batch["sinogram"].to(self.device)  # Measured sinogram
            out, loss = self.model(y, x_gt)  # Predicted volume from the model

            reduced_loss = reduce_loss_dict(loss, self.world_size) # NOTE: we need to reduce the loss across all processes to get the correct loss values for logging and visualization. This is important because each process will have its own local loss values based on its portion of the data, and we want to aggregate these values to get a global view of the evaluation results.

            if self.verbose and self.is_main:

                pbar.set_postfix({
                    key: f"{value.item():.4f}" for key, value in reduced_loss.items()
                })
        
        if self.is_main:

            final = {key: out.cpu().float() for key, out in out.items()} if out is not None else None  # Move the final output to CPU for visualization and saving results. This can be customized based on the requirements of the project (e.g., handling multiple outputs, etc.).
            final["x_gt"] = x_gt.cpu().float()  # Add the ground truth volume to the final output for visualization and saving results. This can be useful for comparing the predicted volume with the ground truth during evaluation, etc.
            final["meta_data"] = batch["meta_data"]  # Add the meta data to the final output for visualization and saving results. This can be useful for analyzing the results in the context of the specific study or sample being processed, etc.

            self._plot_eval(
                x_pred=final["x"] if final is not None else None,
                f_pred=final["f"] if final is not None else None,
                x_gt=final["x_gt"] if final is not None else None,
                y=final["y"] if final is not None else None,
                v=final["v"] if final is not None else None,
                meta_data=final["meta_data"] if final is not None else None,
                epoch="test"
            )

            QM = self._get_quality_measures(
                x_pred=final["x"] if final is not None else None,
                x_gt=final["x_gt"] if final is not None else None,
                v=final["v"] if final is not None else None
            )
            print(f"Quality measures on test set: {QM}")


def main():

    training_config_path = Path("train/train.json")
    if training_config_path.exists():
        with open(training_config_path, "r") as f:
            train_config = json.load(f)
    else:
        print(f"Warning: Training config file not found at {training_config_path}. Using default training configuration.")
        train_config = None
    



    trainer = ParallelHLPDTrainer(train_config=train_config, qualities=["high"])
    try:
        trainer.train()
    finally:
        cleanup_distributed(trainer.distributed)

if __name__ == "__main__":
    main()

    

# DataParallel: 70 s / 8 baches = 8.75 s / batch

# Single GPU: 11.6 s / batch

# DistributedDataParallel 8 batches: 30 s / 8 batches = 3.75 s / batch
# DistributedDataParallel 4 batches: 12 s / 4 batches = 3 s / batch


