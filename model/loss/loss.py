import torch
from torch import nn
from model.loss.quality_measures import helmholtz_norm, ssim, ngf, vgg16_perceptual_loss
from torch.amp import autocast
from pathlib import Path
import json 




class HLPDLoss(nn.Module):

    def __init__(self, weights: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = self._load_weights(weights)

    
    def _load_weights(self, weights: dict):
        """Load the weights for the different loss components. This can be used to customize the relative importance of the different loss components based on the requirements of the project."""
        
        if isinstance(weights, dict):
            return weights
        if isinstance(weights, str):
            # Load weights from a JSON file
            with open(weights, "r") as f:
                weights_dict = json.load(f)
            return weights_dict
        if isinstance(weights, Path):
            # Load weights from a JSON file specified by a Path object
            with open(weights, "r") as f:
                weights_dict = json.load(f)
            return weights_dict
        else:
            weights_path = "model/loss/loss.json"
            if Path(weights_path).exists():
                with open(weights_path, "r") as f:
                    weights_dict = json.load(f)
                return weights_dict
            else:
                raise ValueError(f"Invalid weights input: {weights}. Expected a dictionary, a string path to a JSON file, or a Path object pointing to a JSON file. Also checked for default weights file at {weights_path}, but it was not found. Please provide valid weights input.")

    def _get_image_based_loss(self, x: torch.Tensor, y: torch.Tensor, loss_type: dict[str, float])->torch.Tensor:


        loss_dict = {
            "l2": lambda: torch.mean((x - y) ** 2),
            "l1": lambda: torch.mean(torch.abs(x - y)),
            "ssim": lambda: 1 - ssim(x, y),
            "ngf": lambda: ngf(x, y),
            "vgg16": lambda: vgg16_perceptual_loss(x, y),
        }
        image_loss = {}
        for loss, weight in loss_type.items():
            if loss not in loss_dict:
                raise ValueError(f"Invalid loss type: {loss}. Expected one of {list(loss_dict.keys())}.")
            if weight is not None and weight > 0:
                image_loss[loss] = loss_dict[loss]() * weight
        return image_loss


    def _get_regularization_loss(self, v: torch.Tensor, alpha: float, gamma: float, beta: float, f_pred: torch.Tensor, loss_type: dict[str, float])->torch.Tensor:

        def _TV_loss(x: torch.Tensor)->torch.Tensor:
            """Compute the total variation loss on the spatio-temporal volume. 
            
            NOTE: This calculates TV over both spatial and temporal dimensions, which can help encourage smoothness in both space and time. """
            return torch.mean(torch.abs(x[:, 1:, :, :, :] - x[:, :-1, :, :, :])) + torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])) + torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])) + torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))


        def _helmholtz_loss(v: torch.Tensor, alpha: float, gamma: float, beta: float)->torch.Tensor:
            """Compute the Helmholtz loss on the velocity field."""

            with autocast(device_type="cuda", enabled=False):  # NOTE: helmholtz loss (in particular the FFT implementation) is not compatible with bfloat16 precision. 
                #Hence,  we disable autocast for this part of the loss calculation to ensure that the velocity field is in float32 precision for the Helmholtz loss calculation. 
                v_fp32 = v.float() 
                return helmholtz_norm(v_fp32, alpha, gamma, beta)

        def _drift_loss(v: torch.Tensor, skip_z=True)->torch.Tensor:
            """This """
            dims = (-2, -3, -4)
            v = v[..., 1:] if skip_z else v
            return torch.mean(torch.abs(torch.mean(v, dim=dims))) 

        loss_dict = {
            "helmholtz": lambda:  _helmholtz_loss(v, alpha, gamma, beta),
            "tv": lambda: _TV_loss(f_pred),
            "drift": lambda: _drift_loss(v),
        }
        regularization_loss = {}
        for loss, weight in loss_type.items():
            if loss not in loss_dict:
                raise ValueError(f"Invalid loss type: {loss}. Expected one of {list(loss_dict.keys())}.")
            if weight is not None and weight > 0:
                regularization_loss[loss] = loss_dict[loss]() * weight
        return regularization_loss

    def data_loss(self, x_pred: torch.Tensor, x_gt: torch.Tensor)->torch.Tensor:
        """Compute the data loss between the predicted registration and the ground truth image. """
        data_loss = self._get_image_based_loss(x_pred, x_gt, loss_type=self.weights["image"])  # Compute the image-based loss between the predicted reconstruction and the ground truth image. This can be used to encourage the model to produce reconstructions that are consistent with the ground truth images, which can help improve the overall performance of the model. The specific implementation of this image-based loss can be customized based on the requirements of the project (e.g., using different loss functions, etc.).
        data_loss = {f"data_{k}": v for k, v in data_loss.items()}  # Prefix the data loss keys with "data_" to distinguish them from the consistency loss keys. This can be useful for monitoring the different loss components during training and evaluation. The specific naming convention can be customized based on the requirements of the project.
        return data_loss

    def consistency_loss(self, f_pred: torch.Tensor, x_gt: torch.Tensor)->torch.Tensor:
        concistency_loss = self._get_image_based_loss(f_pred, x_gt, loss_type=self.weights["consistency"])  # Compute the consistency loss between the predicted reconstruction and the ground truth image. This can be used to encourage the model to produce reconstructions that are consistent with the ground truth images, which can help improve the overall performance of the model. The specific implementation of this consistency loss can be customized based on the requirements of the project (e.g., using different loss functions, etc.).
        concistency_loss = {f"consistency_{k}": v for k, v in concistency_loss.items()}  # Prefix the consistency loss keys with "consistency_" to distinguish them from the data loss keys. This can be useful for monitoring the different loss components during training and evaluation. The specific naming convention can be customized based on the requirements of the project.
        return concistency_loss

    def regularization_loss(self, v: torch.Tensor, f_pred: torch.Tensor, alpha: float=1.0, gamma: float=1.0, beta: float=1.0)->torch.Tensor:
        """Compute the regularization loss on the velocity field, which can include a combination of the total variation loss on the predicted reconstruction and the Helmholtz loss on the velocity field. The specific combination of regularization losses can be customized based on the requirements of the project."""
        regularization_loss = self._get_regularization_loss(v, alpha, gamma, beta, f_pred, loss_type=self.weights["regularization"])

        return regularization_loss

    def total_loss(self, x_pred: torch.Tensor, f_pred: torch.Tensor, x_gt: torch.Tensor, v: torch.Tensor, alpha: float=1.0, gamma: float=1.0, beta: float=1.0)->torch.Tensor:
        """Compute the total loss for the HLPD model, which can include a combination of the image L2 loss, image L1 loss, total variation loss on the velocity field, and Helmholtz loss on the velocity field. The specific combination of losses can be customized based on the requirements of the project."""
        data_loss = self.data_loss(x_pred, x_gt)
        consistency_loss = self.consistency_loss(f_pred, x_gt)
        regularization_loss = self.regularization_loss(v, f_pred, alpha, gamma, beta)

        loss = {**data_loss, **consistency_loss, **regularization_loss}  
        total_loss = sum(loss.values())
        loss["total_loss"] = total_loss
        return loss
    
    def forward(
        self,
        x_pred: torch.Tensor,
        f_pred: torch.Tensor,
        x_gt: torch.Tensor,
        v: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute the total loss for the HLPD model, which can include a combination of the image L2 loss, image L1 loss, total variation loss on the velocity field, and Helmholtz loss on the velocity field. The specific combination of losses can be customized based on the requirements of the project."""
        

        loss = self.total_loss(x_pred, f_pred, x_gt, v, alpha, gamma, beta)

        return loss