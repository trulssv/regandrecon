import torch
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

# Model stuff

from model.model import HLPDModel, HLPDModelProfiling, ResidualHLPDModel
from model.loss.loss import HLPDLoss
from model.loss.quality_measures import compute_quality_measures
from data.data_loaders import RegAndReconDataset

# Visualization

from visualization.dynamic_visualization import DynamicVisualization
from matplotlib import pyplot as plt


class HLPDTrainer:

    def __init__(
            self, 
            train_config: dict = None,
            qualities: list[str] = None, 
            verbose: bool = True,
            device: str = None
        )->None:


        self.train_config = train_config
        self.data_root = Path(train_config.get("data_root", "/mnt/data/LDDMM")) if train_config else Path("/mnt/data/LDDMM")
        self.batch_size = train_config.get("batch_size", 1) if train_config else 1
        self.device = device if device else (train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu") if train_config else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose




        # Model and data

        self.model, self.loss_fn = self._init_model_and_loss()
        self.data_loaders =[self._init_data_loader(qualities=qualities, mode=mode) for mode in ["train", "val", "test"]]

        # Optimizer and Scheduler parameters


        self.optimizer = self._init_optimizer(self.model)
        self.scheduler = self._init_scheduler(self.optimizer)

        # Training params

        self.num_epochs = self.train_config["training"].get("num_epochs", 100) if self.train_config else 100
        self.log_interval = self.train_config["training"].get("log_interval", 10) if self.train_config else 10
        self.save_interval = self.train_config["training"].get("save_interval", 10) if self.train_config else 10
        self.checkpoint_dir = Path(self.train_config["training"].get("checkpoint_dir", "./checkpoints")) if self.train_config else Path("./checkpoints")

       


    def _init_model_and_loss(self, profiling: bool = False, parallel: bool = True)->tuple[HLPDModel, HLPDLoss]:

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

        if profiling:
            model = HLPDModelProfiling(lambda_channels=lambda_channels, gamma_channels=gamma_channels, sigma_channels=sigma_channels, hlpd_iterations=hlpd_iterations, dropout=dropout, batch_norm=batch_norm, device=self.device, batch_size=self.batch_size).to(self.device)
        else:
            model = ResidualHLPDModel(lambda_channels=lambda_channels, gamma_channels=gamma_channels, sigma_channels=sigma_channels, hlpd_iterations=hlpd_iterations, dropout=dropout, batch_norm=batch_norm, device=self.device, batch_size=self.batch_size).to(self.device)
        loss_fn = HLPDLoss()

        if parallel and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            model = torch.nn.DataParallel(model)

        # initialize model

        model._initialize()

        return model, loss_fn

    def _init_data_loader(self, qualities: list[str], mode: str)->RegAndReconDataset:

        # NOTE: we only consider the processed volume, the simulated CT data and the meta data
        files = ["volume_processed.pt", "sinogram.pt", "meta_data.json"]

        dataset = RegAndReconDataset(qualities=qualities, mode=mode, data_root=self.data_root, files=files)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return data_loader

    def _init_optimizer(self, model: HLPDModel)->torch.optim.Optimizer:

        learning_rate = self.train_config["optimizer"]["params"].get("lr", 1e-4) if self.train_config else 1e-4
        weight_decay = self.train_config["optimizer"]["params"].get("weight_decay", 0) if self.train_config else 0
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def _init_scheduler(self, optimizer: torch.optim.Optimizer)->torch.optim.lr_scheduler.StepLR:
        step_size = self.train_config["scheduler"]["params"].get("step_size", 50) if self.train_config else 50
        gamma = self.train_config["scheduler"]["params"].get("gamma", 0.8) if self.train_config else 0.8
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler
    
    def _setup_checkpointing(self, checkpoint_dir: Path)->Path:

        """Creates a unique directory for saving checkpoints based on the current date and time. This can be useful for organizing checkpoints from different training runs and avoiding overwriting existing checkpoints. The directory is created under the specified checkpoint directory, and the name of the subdirectory is based on the current date and time in the format "YYYYMMDD_HHMMSS". The function also ensures that the checkpoint directory exists by creating it if it does not already exist."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_checkpoint_dir = checkpoint_dir / time_stamp
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return run_checkpoint_dir



    def _plot_losses(self, losses: dict[str, list[dict[str, float]]])->None:
        """This function can be used to plot the training and validation losses over epochs. This can be useful for monitoring the training progress and diagnosing potential issues with the training process."""
        
        train_losses = losses["train"]
        val_losses = losses["val"]
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 5))

        for key in train_losses[0].keys():
            train_loss_values = [loss[key] for loss in train_losses]
            val_loss_values = [loss[key] if key in loss else None for loss in val_losses]  # Some validation losses might be missing if the validation loop is not fully implemented yet. We can handle this by checking if the key exists in the validation loss dictionary before trying to access it.

            ax1.plot(train_loss_values, label=f"Train {key}")
            ax2.plot(val_loss_values, label=f"Val {key}")
        ax1.set_yscale("log")  # Set y-axis to log scale to better visualize losses that can vary widely in magnitude. This can be customized based on the specific losses being plotted and their expected ranges.
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Losses")
        ax1.legend()
        ax1.grid(True)

        ax2.set_yscale("log")  # Set y-axis to log scale to better visualize losses that can vary widely in magnitude. This can be customized based on the specific losses being plotted and their expected ranges.
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Losses")
        ax2.legend()
        ax2.grid(True)
        
        plt_path = self.run_checkpoint_dir / "visualizations" / f"losses.png"
        print(f"Saving loss plot to {plt_path}")
        plt_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plt_path)
        plt.close()



    def _save_checkpoint(self, epoch: int, model: HLPDModel, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, losses: dict[str, list[float]])->None:
        """This function can be used to save the model checkpoint at regular intervals during training. The checkpoint can include the model state, optimizer state, scheduler state, and any other relevant information such as the current epoch and the training and validation losses. This can be useful for resuming training from a specific checkpoint in case of interruptions, or for analyzing the training process after it has completed."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "losses": losses
        }
        checkpoint_path = self.run_checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def train(self)->None:
        
        # Set up checkpointing directory for this training run

        self.run_checkpoint_dir = self._setup_checkpointing(self.checkpoint_dir)

        losses = {"train": [], "val": []} 
        QM = [] 

        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)
            self.scheduler.step()

            # Save losses for logging

            losses["train"].append(train_loss)
            losses["val"].append(val_loss)

            if (epoch + 1) % self.log_interval == 0:
                self._plot_losses(losses)
                
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, self.model, self.optimizer, self.scheduler, losses)
    

    def _train_one_epoch(self, epoch: int)->dict[str, float]:
        self.model.train()
        train_loader = self.data_loaders[0]  
        
        
        epoch_loss = {}   

        out, final = None, None

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
            # Implement the training logic for one epoch, which typically involves iterating over the batches of data, computing the loss, and updating the model parameters using the optimizer.
            

            x_gt = batch["volume_processed"].to(self.device)  # Ground truth volume
            y = batch["sinogram"].to(self.device)  # Measured sinogram
            out = self.model(y)  # Predicted volume from the model

            loss = self.loss_fn(out["x"], out["f"], x_gt, out["v"])  # Compute the loss

            total_loss = loss["total_loss"]  # Assuming the loss function returns a dictionary with a key "total_loss" that contains the total loss value. This can be customized based on the implementation of the loss function.

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Save losses

            for key, value in loss.items():
                if key not in epoch_loss:
                    epoch_loss[key] = []
                epoch_loss[key].append(value.item())

            if self.verbose:

                for key, value in loss.items():
                    print(f"{key}: {value.item():.4f}", end=" | ")
                print()

            final = {key: out.detach().cpu().float() for key, out in out.items()} if out is not None else None  # Move the final output to CPU for visualization and saving results. This can be customized based on the requirements of the project (e.g., handling multiple outputs, etc.).
            final["x_gt"] = x_gt.cpu().float()  # Add the ground truth volume to the final output for visualization and saving results. This can be useful for comparing the predicted volume with the ground truth during evaluation, etc. 
            final["y"] = y.cpu().float()  # Add the measured sinogram to the final output for visualization and saving results. This can be useful for analyzing the results in the context of the specific measurements being processed, etc. 
            final["meta_data"] = batch["meta_data"]  # Add the meta data to the final output for visualization and saving results. This can be useful for analyzing the results in the context of the specific study or sample being processed, etc.


        return epoch_loss, final


    def _validate(self, epoch: int)->None:
        self.model.eval()
        val_loader = self.data_loaders[1]  # Assuming the second data loader is for validation. This can be customized based on the requirements of the project (e.g., handling multiple data loaders, etc.).
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{self.num_epochs}"):
                # Implement the validation logic for one epoch, which typically involves iterating over the batches of validation data, computing the loss, and optionally saving the model if it achieves a new best performance on the validation set.
                pass
    

    def _plot_eval(self, x_pred: torch.Tensor, f_pred: torch.Tensor, x_gt: torch.Tensor, v: torch.Tensor, y: torch.Tensor, meta_data: dict, epoch: int, mode="val")->None:
        """This function can be used to plot the evaluation results during training, such as the training and validation losses over epochs, or any other relevant metrics. This can be useful for monitoring the training progress and diagnosing potential issues with the training process."""

        # Transfer tensors to CPU, if this is not already done.

        x_pred = x_pred.cpu()
        f_pred = f_pred.cpu()
        x_gt = x_gt.cpu()
        v = v.cpu()
        y = y.cpu()


        plane = "axial"  # This can be customized based on the requirements of the project (e.g., visualizing different planes, etc.). The specific implementation of the visualization can also be customized based on the requirements of the project (e.g., using different visualization techniques, etc.).

        visualization_dir = self.run_checkpoint_dir / "visualizations" / f"epoch_{epoch}"
        visualization_dir.mkdir(parents=True, exist_ok=True)

        # Visualize v
        vis = DynamicVisualization(meta_data=meta_data, vmin=0.0, vmax=2.0, save_dir=visualization_dir)
        vis.plot_velocity_field(v, plane=plane, time_bin=0, prefix=f"{mode}_velocity_field")
        for label, x in {"Registred Volume": x_pred, "Reconstructed Volume": f_pred, "Ground Truth Volume": x_gt}.items(): 
            prefix = label.lower().replace(" ", "_")
            vis.visualize_plane_dynamic(x, plane=plane, slice_idx=None, prefix=f"{mode}_{prefix}")
            if label is not "Registred Volume": # The registered and reconstructed volumes coincide for t=0. 
                vis.visualize_slices(x, plane=plane, time_bin=0, prefix=f"{mode}_{prefix}")
        
        vis = DynamicVisualization(meta_data=meta_data, vmin=y.min().item(), vmax=y.max().item(), save_dir=visualization_dir)
        vis.visualize_plane_dynamic(y, plane=plane, slice_idx=None, prefix=f"{mode}_simulated_sinogram")
        vis.visualize_slices(y, plane=plane, time_bin=0, prefix=f"{mode}_measured_sinogram")


    def _get_quality_measures(self, x_pred: torch.Tensor, x_gt: torch.Tensor, v)->dict[str, float]:
        """This function can be used to compute quality measures for the predicted volume compared to the ground truth volume. This can be useful for evaluating the performance of the model and monitoring the training progress. The specific quality measures to be computed can be customized based on the requirements of the project (e.g., mean squared error, structural similarity index, etc.)."""
        
        x_pred = x_pred.cpu()
        x_gt = x_gt.cpu()
        v = v.cpu()

        include_perceptual = False
        if self.train_config is not None:
            include_perceptual = self.train_config.get("training", {}).get("include_perceptual_quality", False)

        QM = compute_quality_measures(x_pred, x_gt, v, include_perceptual=include_perceptual)
        print("Quality Measures: ", end="")
        for key, value in QM.items():
            print(f"{key}: {value:.4f}", end=" | ")
        print("\n")
        
        return QM


class SingleSampleHLPDTrainer(HLPDTrainer):
    """This class overfits an HLPD model to a single sample from the training set. This can be useful for debugging and analyzing the behavior of the model on a specific sample, as well as for visualizing the training process in more detail. The specific implementation of this class can be customized based on the requirements of the project (e.g., handling different types of data, etc.)."""



    def __init__(self, train_config: dict = None, qualities: list[str] = None, verbose: bool = True, device: str = None)->None:
        super().__init__(train_config=train_config, qualities=qualities, verbose=verbose, device=device)

        # Overwrite the mode specific data loaders with a single sample data loader for overfitting. This can be customized based on the requirements of the project (e.g., handling multiple samples, etc.).
        
        self.data_loaders = [self._init_data_loader(qualities=qualities, mode="train")]  # Assuming we want to overfit on a single sample from the training set. This can be customized based on the requirements of the project (e.g., handling multiple samples, etc.).



    def _init_data_loader(self, qualities: list[str], mode: str)->RegAndReconDataset:

        # NOTE: we only consider the processed volume, the simulated CT data and the meta data
        files = ["volume_processed.pt", "sinogram.pt", "meta_data.json"]
        dataset = RegAndReconDataset(qualities=qualities, mode=mode, data_root=self.data_root, files=files) 

        # We only take a single sample from the dataset for overfitting. This can be customized based on the requirements of the project (e.g., handling multiple samples, etc.).
        single_sample_dataset = torch.utils.data.Subset(dataset, indices=[0])  # Assuming we want to take the first sample from the dataset. This can be customized based on the requirements of the project (e.g., selecting a specific sample based on certain criteria, etc.).
        data_loader = torch.utils.data.DataLoader(single_sample_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return data_loader  
    

    def train(self):
        """We overwrite the train method to ensure train and val uses the same single sample data loader for overfitting. This can be customized based on the requirements of the project (e.g., handling multiple samples, etc.)."""


        # Set up checkpointing directory for this training run

        self.run_checkpoint_dir = self._setup_checkpointing(self.checkpoint_dir)

        losses = {"train": [], "val": []}

        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
            loss, out = self._train_one_epoch(epoch)
            self.scheduler.step()

            # Save losses for logging

            losses["train"].append(loss)
            losses["val"].append(loss)  # We use the same loss for validation since we are overfitting on a single sample. This can be customized based on the requirements of the project (e.g., handling multiple samples, etc.).


            if (epoch + 1) % self.log_interval == 0:
                self._plot_eval(
                    x_pred=out["x"] if out is not None else None,
                    f_pred=out["f"] if out is not None else None,
                    x_gt=out["x_gt"] if out is not None else None,
                    y=out["y"] if out is not None else None,
                    v=out["v"] if out is not None else None,
                    meta_data=out["meta_data"] if out is not None else None,
                    epoch=epoch+1

                )
                self._plot_losses(losses)
                
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, self.model, self.optimizer, self.scheduler, losses)
    

def main():

    training_config_path = Path("train/train.json")
    if training_config_path.exists():
        with open(training_config_path, "r") as f:
            train_config = json.load(f)
    else:
        print(f"Warning: Training config file not found at {training_config_path}. Using default training configuration.")
        train_config = None
    

    trainer = SingleSampleHLPDTrainer(train_config=train_config, qualities=["high"])
    trainer.train()

if __name__ == "__main__":
    main()