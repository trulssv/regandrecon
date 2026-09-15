import torch
from operators.lddmm.helmholtz import HelmholtzOperator
from functools import lru_cache


def helmholtz_norm(v: torch.Tensor, alpha: float, gamma: float, beta: float)->torch.Tensor:
    """Compute the Helmholtz norm on the velocity field."""
    helmholtz_op = HelmholtzOperator(alpha, gamma, beta)
    v_reg = helmholtz_op(v)
    return torch.mean(v_reg * v)

def mse(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
    """Compute the mean squared error between two images."""
    return torch.mean((x - y) ** 2)

def ssim(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
    """Compute the structural similarity index between two images. This is done per time bin"""
    
    def _mean(x: torch.Tensor)->torch.Tensor:
        return torch.mean(x)
    
    def _variance(x: torch.Tensor)->torch.Tensor:
        return torch.var(x)

    def _covariance(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
        return torch.mean((x - _mean(x)) * (y - _mean(y)))


    assert x.shape == y.shape, f"Expected input images to have the same shape, but got {x.shape} and {y.shape}. Please check the input format."
    assert x.dim() == 5, f"Expected input images to have 5 dimensions (B x T x D x H x W), but got {x.shape}. Please check the input format."
    

    ssim = 0.0

    for t in range(x.shape[1]):
        x_t = x[:, t, :, :, :]
        y_t = y[:, t, :, :, :]


        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = _mean(x_t) 
        mu_y = _mean(y_t)
        sigma_x = _variance(x_t)
        sigma_y = _variance(y_t)
        sigma_xy = _covariance(x_t, y_t)
        ssim += ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim / x.shape[1]

def ngf(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-3)->torch.Tensor:
    """Compute the normalized gradient field between two images. The parameter epsilon controls the sensitivity of the NGF to small gradients, which can help improve the stability of the NGF in cases where the images have low contrast or are noisy.
    For optimal performance, epsilon should be set to roughly the noise level in the images.
    """
    def _gradient(x: torch.Tensor)->torch.Tensor:
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)
        grad_z = torch.zeros_like(x)
        grad_x[..., 1:, :, :] = x[..., 1:, :, :] - x[..., :-1, :, :]
        grad_y[..., :, 1:, :] = x[..., :, 1:, :] - x[..., :, :-1, :]
        grad_z[..., :, :, 1:] = x[..., :, :, 1:] - x[..., :, :, :-1]
        return torch.stack((grad_x, grad_y, grad_z), dim=-1)
    
    grad_x = _gradient(x)
    grad_y = _gradient(y)
    ngf = torch.mean((grad_x - grad_y) ** 2) / (torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2) + epsilon)
    return ngf

@lru_cache(maxsize=None)
def _get_vgg16_cached(device: torch.device)->torch.nn.Module:
    """Get the VGG16 feature extractor for computing the perceptual loss. We use lru_cache to ensure that we only load the VGG16 model once, which can help improve the efficiency of the training process."""
    from torchvision import models
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(device)
    for param in m.parameters():
        param.requires_grad = False
    return m 

def get_vgg16_features(device: torch.device)->torch.nn.Module:
    """Get the VGG16 feature extractor for computing the perceptual loss. We use lru_cache to ensure that we only load the VGG16 model once, which can help improve the efficiency of the training process."""
    return _get_vgg16_cached(device)


def vgg16_perceptual_loss(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
    """Compute the VGG16 perceptual loss between two images. """

    def _normalize(x: torch.Tensor)->torch.Tensor:
        """Normalizes the input tensor to roughly be in the range [-1, 1] (which VGG16 expects). Since the volumes are in attenuation water equivalents, they mostly in the range [0, 2], so we just subtract 1 to center around zero. """
        return x - 1.0
    
    assert x.device == y.device, f"Expected input images to be on the same device, but got {x.device} and {y.device}. Please check the input format."
    assert x.shape == y.shape, f"Expected input images to have the same shape, but got {x.shape} and {y.shape}. Please check the input format."
    assert x.dim() == 5, f"Expected input images to have 5 dimensions (B x T x D x H x W), but got {x.shape}. Please check the input format."
    
    vgg16 = get_vgg16_features(x.device)

    B, T, D, H, W = x.shape

    # Normalize the input images to roughly be in the range [-1, 1], which is what VGG16 expects. Since the volumes are in attenuation water equivalents, they mostly in the range [0, 2], so we just subtract 1 to center around zero.

    x, y = _normalize(x), _normalize(y)




    def _extract_slice(x: torch.Tensor, plane:str, time_bin:int) -> torch.Tensor:
        plane_to_dim = {"axial": 0, "coronal": 1, "sagittal": 2}
        dim = plane_to_dim[plane]
        if dim == 0:
            slice_idx = H // 2
            return x[:, time_bin, :, slice_idx, :]
        elif dim == 1:
            slice_idx = W // 2
            return x[:, time_bin, :, :, slice_idx]
        elif dim == 2:
            slice_idx = D // 2
            return x[:, time_bin, slice_idx, :, :]
        else:
            raise ValueError(f"Invalid plane: {plane}. Expected one of 'axial', 'coronal', or 'sagittal'.")

    def _extract_features(x: torch.Tensor)->torch.Tensor:
        assert x.dim() == 3, f"Expected input slice to have 3 dimensions (B x H x W), but got {x.shape}. Please check the input format."
        # Convert the input slice to 3 channels by repeating it along the channel dimension, since VGG16 expects 3-channel input.
        x = x.unsqueeze(1).repeat(1, 3, 1, 1) # Shape: (B, 3, H, W)
        return vgg16(x)
    
    features_all = []
    for plane in ["axial", "coronal", "sagittal"]:
        for time_bin in range(T):
            x_slice = _extract_slice(x, plane, time_bin=time_bin)
            y_slice = _extract_slice(y, plane, time_bin=time_bin)
            features_x = _extract_features(x_slice)
            features_y = _extract_features(y_slice)
            features_all.append((features_x, features_y))
    
        
    perceptual_loss = 0.0
    for features_x, features_y in features_all:
        perceptual_loss += torch.mean((features_x - features_y) ** 2) 

    return perceptual_loss / len(features_all)   
    
def compute_quality_measures(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    v: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 1.0,
    beta: float = 1.0,
    include_perceptual: bool = False,
) -> dict[str, float]:
    """Compute quality measures for the predicted volume compared to the ground truth volume. This can be useful for evaluating the performance of the model and monitoring the training progress. The specific quality measures to be computed can be customized based on the requirements of the project (e.g., mean squared error, structural similarity index, etc.)."""
    mse_val = mse(x_pred, x_gt).item()
    ssim_val = ssim(x_pred, x_gt).item()
    ngf_val = ngf(x_pred, x_gt).item()
    helmholtz_val = helmholtz_norm(v, alpha, gamma, beta).item()

    quality_measures = {
        "mse": mse_val,
        "ssim": ssim_val,
        "ngf": ngf_val,
        "helmholtz": helmholtz_val,
    }

    if include_perceptual:
        quality_measures["perceptual_loss"] = vgg16_perceptual_loss(x_pred, x_gt).item()

    return quality_measures