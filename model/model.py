import torch
from torch import nn
from model.blocks import GammaBlock, LambdaBlock, SigmaBlock, ResidualGammaBlock, ResidualLambdaBlock, ResidualFiLMGammaBlock
from model.loss.loss import HLPDLoss

from pathlib import Path
import json

# Operators

from operators.ray_transform import DynamicRayTransform
from operators.lddmm.deform import FlowDeformationOperator



class HLPDModel(nn.Module):
    """This class implements the HLPD model, which is a deep learning-based approach for motion compensation in medical imaging. 
    The HLPD model consists of a CNN module that estimates the velocity field from the input image, and a deformation module that applies the estimated velocity field to the input image to produce the deformed image. 
    The HLPD model is trained end-to-end using a loss function that combines a data fidelity term and a regularization term on the velocity field."""
    def __init__(
            self, 
            dropout: float = 0.0, 
            batch_norm: bool = True, 
            lambda_channels: list[int] = None, 
            gamma_channels: list[int] = None, 
            sigma_channels: list[int] = None,
            hlpd_iterations: int = 10,
            batch_size: int = 1,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )->None:
        

        super().__init__()
        self.batch_size = batch_size
        self.device = device

        # Module parameters

        self.lambda_channels = lambda_channels
        self.gamma_channels = gamma_channels
        self.sigma_channels = sigma_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.hlpd_iterations = hlpd_iterations

        # Load geoemetry information from the meta data, such as the dimensions of the input image, the spacing, and any other relevant parameters. The geometry information can be used to initialize the operators and components of the HLPD model.

        self.geometry = self._get_geometry()
        self.T = self.geometry.get("time_bins", None)



    def _initialize(self):
        # Initialize the components of the HLPD model, such as the CNN module for estimating the velocity field, and any other relevant components. The initialization can be based on the dimensions of the input image and any other relevant parameters.

        self.lambda_nets = torch.nn.ModuleList([LambdaBlock(self.T, self.lambda_channels, dropout=self.dropout, batch_norm=self.batch_norm).to(self.device) if self.lambda_channels is not None else None for _ in range(self.hlpd_iterations)])
        self.gamma_nets = torch.nn.ModuleList([GammaBlock(self.T, self.gamma_channels, dropout=self.dropout, batch_norm=self.batch_norm).to(self.device) if self.gamma_channels is not None else None for _ in range(self.hlpd_iterations)])
        self.sigma_nets = torch.nn.ModuleList([SigmaBlock(self.T, self.sigma_channels, dropout=self.dropout, batch_norm=self.batch_norm).to(self.device) if self.sigma_channels is not None else None for _ in range(self.hlpd_iterations)])

        # Initialize the operators used in the HLPD model, such as the ray transform and the deformation operator. The operators are typically initialized based on the dimensions of the input image and any other relevant parameters.

        self.ray_trafo, self.flow_deform = self._initialize_operators()

        self.theta = torch.tensor(self.ray_trafo.angles, device=self.device) if self.ray_trafo.angles is not None else None

        
    def _initialize_operators(self) -> tuple[DynamicRayTransform, FlowDeformationOperator]:
        """This function initializes the operators used in the HLPD model, such as the ray transform and the deformation operator. 
        The operators are typically initialized based on the dimensions of the input image and any other relevant parameters."""
       
        run_config_path = Path("data/data_pipeline_run_config.json")

        assert run_config_path.exists(), FileNotFoundError(f"Run config file not found at {run_config_path}. Please run the data pipeline first to generate the run config file.")

        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        

        # NOTE: Meta data can be nested in the run config file, so we need to unnest it to extract the relevant information for initializing the operators. The unnesting can be done using a helper function that recursively flattens the nested structure of the meta data. This is necessary because the meta data may contain information about the dimensions of the input image, the spacing, and any other relevant parameters that are needed to initialize the operators used in the HLPD model.

        from data.utils import _unnest_meta_data
        meta_data = run_config.get("meta_data", {})
        meta_data = _unnest_meta_data(meta_data)    

        ray_trafo_cfg = run_config.get("ray_trafo_cfg", {})
        preprocess_cfg = run_config.get("preprocess_cfg", {})

        data_shape = preprocess_cfg.get("SHAPE")

        ray_trafo = DynamicRayTransform(ray_trafo_cfg=ray_trafo_cfg, preprocess_cfg=preprocess_cfg, meta_data=meta_data)
        flow_deform = FlowDeformationOperator(shape=data_shape, device=self.device)


        # Here we would implement the logic for initializing the operators used in the HLPD model, which typically involves creating instances of the relevant operator classes and configuring them based on the dimensions of the input image and any other relevant parameters.
        return ray_trafo, flow_deform
    
    def _get_geometry(self)->dict:
        """This function extracts the relevant geometry information from the meta data, such as the dimensions of the input image, the spacing, and any other relevant parameters. 
        The geometry information can be used to initialize the operators and components of the HLPD model."""

        run_config_path = Path("data/data_pipeline_run_config.json")
        model_config_path = Path("model/model.json")
        ray_trafo_cfg_path = Path("operators/ray_trafo.json")
        assert run_config_path.exists(), FileNotFoundError(f"Run config file not found at {run_config_path}. Please run the data pipeline first to generate the run config file.")
        assert model_config_path.exists(), FileNotFoundError(f"Model config file not found at {model_config_path}. Please run the data pipeline first to generate the model config file.")
        assert ray_trafo_cfg_path.exists(), FileNotFoundError(f"Ray transform config file not found at {ray_trafo_cfg_path}. Please run the data pipeline first to generate the ray transform config file.")

        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        with open(ray_trafo_cfg_path, "r") as f:
            ray_trafo_cfg = json.load(f)
        
        preprocess_cfg = run_config.get("preprocess_cfg", {})
        data_shape = preprocess_cfg.get("SHAPE", None)
        time_bins = preprocess_cfg.get("time_bins", None)

        detector_shape = ray_trafo_cfg.get("detector_shape", None)
        num_projections_per_time_bin = ray_trafo_cfg.get("dynamic", {}).get("num_projections_per_time_bin", None)



        return {"data_shape": data_shape, "time_bins": time_bins, "detector_shape": detector_shape, "num_projections_per_time_bin": num_projections_per_time_bin}


    def _initialize_hlpd(self, B: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """This function initializes the components of the HLPD model, such as the CNN module for estimating the velocity field, and any other relevant components. 
        The initialization can be based on the dimensions of the input image and any other relevant parameters."""
    

        input_shape = self.geometry.get("data_shape", None)
        time_bins = self.geometry.get("time_bins", None)

        detector_shape = self.geometry.get("detector_shape", None)
        num_projections_per_time_bin = self.geometry.get("num_projections_per_time_bin", None)

        f_shape = (B, time_bins, *input_shape) if input_shape is not None and time_bins is not None and B is not None else None
        h_shape = (B, time_bins, num_projections_per_time_bin, *detector_shape) if detector_shape is not None and num_projections_per_time_bin is not None and B is not None else None

        f0 = torch.zeros(f_shape, device=self.device) if f_shape is not None else None
        h0 = torch.zeros(h_shape, device=self.device) if h_shape is not None else None

        return f0, h0


    def hlpd_iteration(
            self, 
            f: torch.Tensor, 
            h: torch.Tensor, 
            y: torch.Tensor, 
            x: torch.Tensor, 
            i: int, 
            amp_enabled: bool=True, 
            amp_dtype: torch.dtype=torch.bfloat16, 
            ray_trafo_in_fp32: bool=True, 
            deform_in_fp32: bool=True,
            profiling: bool=True
        )->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if amp_enabled:
            # Forward projection

            if ray_trafo_in_fp32:
                with torch.autocast(device_type="cuda", enabled=False):
                    Tf = self.ray_trafo(f.float())
            else:
                with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                    Tf = self.ray_trafo(f)

            # Gamma net

            with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                h = self.gamma_nets[i](h, Tf, y) 

            # Backprojection
            
            if ray_trafo_in_fp32:
                with torch.autocast(device_type="cuda", enabled=False):
                    TTh = self.ray_trafo.backproject(h.float())
            else:
                with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                    TTh = self.ray_trafo.backproject(h)

            # Lambda net
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                f = self.lambda_nets[i](f, TTh, x)

            # Sigma net

            
            with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                v = self.sigma_nets[i](x, f) 

            # Flow Deformation Operator
            
            if deform_in_fp32:
                with torch.autocast(device_type="cuda", enabled=False):
                    x = self.flow_deform(f[:, [0], ...].float(), v.float()) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).
            else:
                with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                    x = self.flow_deform(f[:, [0], ...], v) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).        
        else:
            # Forward projection

            Tf = self.ray_trafo(f)

            # Gamma net

            h = self.gamma_nets[i](h, Tf, y) 

            # Backprojection
            
            TTh = self.ray_trafo.backproject(h)
            

            # Lambda net
            
            f = self.lambda_nets[i](f, TTh, x)

            # Sigma net

            
            v = self.sigma_nets[i](x, f) 

            # Flow Deformation Operator
            
            x = self.flow_deform(f[:, [0], ...] , v) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).


        return f, h, x, v


    def forward(self, y: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        B = y.shape[0]
        f, h = self._initialize_hlpd(B, y.device)  # NOTE: in _initialize_hlpd we also update the device attribute and reinitialize the operators with the updated device, so we can be sure that all components of the model are on the correct device for the forward pass.
        x = torch.zeros_like(f).to(self.device)

        # AMP logic

        amp_enabled = False
        amp_dtype = torch.bfloat16 if amp_enabled else None
        ray_trafo_in_fp32 = True # NOTE: This is currently the only supported option
        deform_in_fp32 = True

        for i in range(self.hlpd_iterations):
            f, h, x, v = checkpoint(
                lambda f_, h_, y_, x_, i_: self.hlpd_iteration(f_, h_, y_, x_, i_, amp_enabled=amp_enabled, amp_dtype=amp_dtype, ray_trafo_in_fp32=ray_trafo_in_fp32, deform_in_fp32=deform_in_fp32, profiling=True),
                f, h, y, x, i,
                use_reentrant=False
            )

        out ={"x": x, "f": f, "v": v} 
        return out


class HLPDModelWithLoss(HLPDModel):
    """This class implements the HLPD model with an integrated loss function, which allows for end-to-end training of the model. The loss function can be defined based on the requirements of the project, and can include a combination of image fidelity terms (e.g., L2 loss, L1 loss) and regularization terms on the velocity field (e.g., total variation loss, Helmholtz loss). The forward method of this class can be modified to compute the loss components in addition to the output of the HLPD model, and return both the output and the loss components as a dictionary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = HLPDLoss()


    def forward(self, y: torch.Tensor, x_gt: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        out = super().forward(y)
        x_pred = out["x"]
        f_pred = out["f"]
        v = out["v"]

        loss = self.loss_fn(x_pred, f_pred, x_gt, v) # Aggregate over the batch dimension to get a single scalar loss value for the entire batch. The specific aggregation method can be customized based on the requirements of the project (e.g., using a different reduction method, etc.). The loss function can also return a dictionary of individual loss components (e.g., L2 loss, L1 loss, total variation loss, Helmholtz loss, etc.) in addition to the total loss, which can be useful for monitoring the training process and diagnosing any issues with the model.

        return out, loss


class ResidualHLPDModel(HLPDModel):
    """This class implements a residual version of the HLPD model, where the output of each iteration is added to the input of the next iteration. This can help to improve the convergence and stability of the model during training, as it allows for a more gradual refinement of the velocity field and the deformed image over multiple iterations. The forward method of this class can be modified to implement the residual connections between iterations, and to return the final output after all iterations are completed."""
    def __init__(
        self, 
        dropout: float = 0.0, 
        batch_norm: bool = True, 
        lambda_channels: list[int] = None, 
        gamma_channels: list[int] = None, 
        sigma_channels: list[int] = None,
        hlpd_iterations: int = 10,
        batch_size: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tau=0.1, 
        sigma=0.1,
        rho=None
        )->None:
    
        super().__init__(dropout, batch_norm, lambda_channels, gamma_channels, sigma_channels, hlpd_iterations, batch_size, device)

        self.lambda_nets = torch.nn.ModuleList([ResidualLambdaBlock(self.T, lambda_channels, dropout=dropout, batch_norm=batch_norm, tau=tau).to(self.device) if lambda_channels is not None else None for _ in range(hlpd_iterations)])
        self.gamma_nets = torch.nn.ModuleList([ResidualFiLMGammaBlock(self.T, gamma_channels, dropout=dropout, batch_norm=batch_norm, tau=sigma).to(self.device) if gamma_channels is not None else None for _ in range(hlpd_iterations)])
        self.rho = rho

    def hlpd_iteration(self, f, h, y, x, v_prev, i, theta):
        

        Tf = self.ray_trafo(f)

        # Gamma net

        h = self.gamma_nets[i](h, Tf, y, theta) 

        # Backprojection
        
        TTh = self.ray_trafo.backproject(h)
        

        # Lambda net
        
        f = self.lambda_nets[i](f, TTh, x)

        # Sigma net

        v = v_prev + self.rho * self.sigma_nets[i](x, f) if v_prev is not None and self.rho is not None else self.sigma_nets[i](x, f)

        # Flow Deformation Operator
        
        x = self.flow_deform(f[:, [0], ...] , v) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).


        return f, h, x, v
    
    def forward(self, y: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        B = y.shape[0]
        f, h = self._initialize_hlpd(B, y.device)  # NOTE: in _initialize_hlpd we also update the device attribute and reinitialize the operators with the updated device, so we can be sure that all components of the model are on the correct device for the forward pass.
        x = torch.zeros_like(f).to(self.device)
        v = None

        # Ensure theta has shaoe B, T

        theta = self.theta.repeat(B, 1).to(self.device) if self.theta is not None else None

        for i in range(self.hlpd_iterations):
            f, h, x, v = checkpoint(
                lambda f_, h_, y_, x_, v_, i_, theta_: self.hlpd_iteration(f_, h_, y_, x_, v_, i_, theta_),
                f, h, y, x, v, i, theta,
                use_reentrant=False
            )

        out ={"x": x, "f": f, "v": v} 
        return out



class ResidualHLPDModelWithLoss(ResidualHLPDModel):
    """This class implements the residual HLPD model with an integrated loss function, which allows for end-to-end training of the model. The loss function can be defined based on the requirements of the project, and can include a combination of image fidelity terms (e.g., L2 loss, L1 loss) and regularization terms on the velocity field (e.g., total variation loss, Helmholtz loss). The forward method of this class can be modified to compute the loss components in addition to the output of the residual HLPD model, and return both the output and the loss components as a dictionary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = HLPDLoss()

    def forward(self, y: torch.Tensor, x_gt: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        out = super().forward(y)
        x_pred = out["x"]
        f_pred = out["f"]
        v = out["v"]

        loss = self.loss_fn(x_pred, f_pred, x_gt, v) # Aggregate over the batch dimension to get a single scalar loss value for the entire batch. The specific aggregation method can be customized based on the requirements of the project (e.g., using a different reduction method, etc.). The loss function can also return a dictionary of individual loss components (e.g., L2 loss, L1 loss, total variation loss, Helmholtz loss, etc.) in addition to the total loss, which can be useful for monitoring the training process and diagnosing any issues with the model.

        return out, loss

from torch.cuda.memory import memory_allocated, max_memory_allocated, reset_peak_memory_stats, memory_reserved, max_memory_reserved, memory_summary
from torch.utils.checkpoint import checkpoint

class HLPDModelProfiling(HLPDModel):
    """This class implements the HLPD model, which is a deep learning-based approach for motion compensation in medical imaging. 
    The HLPD model consists of a CNN module that estimates the velocity field from the input image, and a deformation module that applies the estimated velocity field to the input image to produce the deformed image. 
    The HLPD model is trained end-to-end using a loss function that combines a data fidelity term and a regularization term on the velocity field."""
    def __init__(
            self, 
            dropout: float = 0.0, 
            batch_norm: bool = True, 
            lambda_channels: list[int] = None, 
            gamma_channels: list[int] = None, 
            sigma_channels: list[int] = None,
            hlpd_iterations: int = 10,
            batch_size: int = 1,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )->None:
        

        super().__init__()


        # Import requireed modules for memory profiling and benchmarking

        
        self.profiling = {
            "memory_allocated": [],
            }

        self.batch_size = batch_size
        self.device = device


        self.profiling["memory_allocated"].append(memory_allocated(self.device))  

        # Load geoemetry information from the meta data, such as the dimensions of the input image, the spacing, and any other relevant parameters. The geometry information can be used to initialize the operators and components of the HLPD model.

        self.geometry = self._get_geometry()
        T = self.geometry.get("time_bins", None)

        # Initialize the components of the HLPD model, such as the CNN module for estimating the velocity field, and any other relevant components. The initialization can be based on the dimensions of the input image and any other relevant parameters.

        self.lambda_nets = torch.nn.ModuleList([LambdaBlock(T, lambda_channels, dropout=dropout, batch_norm=batch_norm).to(self.device) if lambda_channels is not None else None for _ in range(hlpd_iterations)])
        self.gamma_nets = torch.nn.ModuleList([GammaBlock(T, gamma_channels, dropout=dropout, batch_norm=batch_norm).to(self.device) if gamma_channels is not None else None for _ in range(hlpd_iterations)])
        self.sigma_nets = torch.nn.ModuleList([SigmaBlock(T, sigma_channels, dropout=dropout, batch_norm=batch_norm).to(self.device) if sigma_channels is not None else None for _ in range(hlpd_iterations)])
        self.hlpd_iterations = hlpd_iterations

        # Initialize the operators used in the HLPD model, such as the ray transform and the deformation operator. The operators are typically initialized based on the dimensions of the input image and any other relevant parameters.

        self.ray_trafo, self.flow_deform = self._initialize_operators()


        
    def _initialize_operators(self) -> tuple[DynamicRayTransform, FlowDeformationOperator]:
        """This function initializes the operators used in the HLPD model, such as the ray transform and the deformation operator. 
        The operators are typically initialized based on the dimensions of the input image and any other relevant parameters."""
       
        run_config_path = Path("data/data_pipeline_run_config.json")

        assert run_config_path.exists(), FileNotFoundError(f"Run config file not found at {run_config_path}. Please run the data pipeline first to generate the run config file.")

        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        

        # NOTE: Meta data can be nested in the run config file, so we need to unnest it to extract the relevant information for initializing the operators. The unnesting can be done using a helper function that recursively flattens the nested structure of the meta data. This is necessary because the meta data may contain information about the dimensions of the input image, the spacing, and any other relevant parameters that are needed to initialize the operators used in the HLPD model.

        from data.utils import _unnest_meta_data
        meta_data = run_config.get("meta_data", {})
        meta_data = _unnest_meta_data(meta_data)    

        ray_trafo_cfg = run_config.get("ray_trafo_cfg", {})
        preprocess_cfg = run_config.get("preprocess_cfg", {})

        data_shape = preprocess_cfg.get("SHAPE")

        ray_trafo = DynamicRayTransform(ray_trafo_cfg=ray_trafo_cfg, preprocess_cfg=preprocess_cfg, meta_data=meta_data)
        flow_deform = FlowDeformationOperator(shape=data_shape, device=self.device)


        # Here we would implement the logic for initializing the operators used in the HLPD model, which typically involves creating instances of the relevant operator classes and configuring them based on the dimensions of the input image and any other relevant parameters.
        return ray_trafo, flow_deform
    
    def _get_geometry(self)->dict:
        """This function extracts the relevant geometry information from the meta data, such as the dimensions of the input image, the spacing, and any other relevant parameters. 
        The geometry information can be used to initialize the operators and components of the HLPD model."""

        run_config_path = Path("data/data_pipeline_run_config.json")
        model_config_path = Path("model/model.json")
        ray_trafo_cfg_path = Path("operators/ray_trafo.json")
        assert run_config_path.exists(), FileNotFoundError(f"Run config file not found at {run_config_path}. Please run the data pipeline first to generate the run config file.")
        assert model_config_path.exists(), FileNotFoundError(f"Model config file not found at {model_config_path}. Please run the data pipeline first to generate the model config file.")
        assert ray_trafo_cfg_path.exists(), FileNotFoundError(f"Ray transform config file not found at {ray_trafo_cfg_path}. Please run the data pipeline first to generate the ray transform config file.")

        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        with open(ray_trafo_cfg_path, "r") as f:
            ray_trafo_cfg = json.load(f)
        
        preprocess_cfg = run_config.get("preprocess_cfg", {})
        data_shape = preprocess_cfg.get("SHAPE", None)
        time_bins = preprocess_cfg.get("time_bins", None)

        detector_shape = ray_trafo_cfg.get("detector_shape", None)
        num_projections_per_time_bin = ray_trafo_cfg.get("dynamic", {}).get("num_projections_per_time_bin", None)



        return {"data_shape": data_shape, "time_bins": time_bins, "detector_shape": detector_shape, "num_projections_per_time_bin": num_projections_per_time_bin}


    def _initialize_hlpd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """This function initializes the components of the HLPD model, such as the CNN module for estimating the velocity field, and any other relevant components. 
        The initialization can be based on the dimensions of the input image and any other relevant parameters."""
        
        input_shape = self.geometry.get("data_shape", None)
        d = len(input_shape) if input_shape is not None else None
        time_bins = self.geometry.get("time_bins", None)

        detector_shape = self.geometry.get("detector_shape", None)
        num_projections_per_time_bin = self.geometry.get("num_projections_per_time_bin", None)

        f_shape = (self.batch_size, time_bins, *input_shape) if input_shape is not None and time_bins is not None and self.batch_size is not None else None
        h_shape = (self.batch_size, time_bins, num_projections_per_time_bin, *detector_shape) if detector_shape is not None and num_projections_per_time_bin is not None and self.batch_size is not None else None

        f0 = torch.zeros(f_shape).to(self.device) if f_shape is not None else None
        h0 = torch.zeros(h_shape).to(self.device) if h_shape is not None else None

        return f0, h0
    
    def hlpd_iteration(
            self, 
            f: torch.Tensor, 
            h: torch.Tensor, 
            y: torch.Tensor, 
            x: torch.Tensor, 
            i: int, 
            amp_enabled: bool=True, 
            amp_dtype: torch.dtype=torch.bfloat16, 
            ray_trafo_in_fp32: bool=True, 
            deform_in_fp32: bool=True,
            profiling: bool=True
        )->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if amp_enabled:
            # Forward projection

            if ray_trafo_in_fp32:
                with torch.autocast(device_type="cuda", enabled=False):
                    Tf = self.ray_trafo(f.float())
            else:
                with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                    Tf = self.ray_trafo(f)
            if profiling:
                print(f"After ray transform at HLPD iteration {i+1} / {self.hlpd_iterations}, Memory Allocated: {memory_allocated(self.device) / 1e9:.2f} GB")

            # Gamma net

            with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                h = self.gamma_nets[i](h, Tf, y) 
            if profiling:
                print(f"After gamma net at HLPD iteration {i+1} / {self.hlpd_iterations}, Memory Allocated: {memory_allocated(self.device) / 1e9:.2f} GB")

            # Backprojection
            
            if ray_trafo_in_fp32:
                with torch.autocast(device_type="cuda", enabled=False):
                    TTh = self.ray_trafo.backproject(h.float())
            else:
                with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                    TTh = self.ray_trafo.backproject(h)
            if profiling:
                print(f"After ray backprojection at HLPD iteration {i+1} / {self.hlpd_iterations}, Memory Allocated: {memory_allocated(self.device) / 1e9:.2f} GB")
            

            # Lambda net
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                f = self.lambda_nets[i](f, TTh, x)
            if profiling:
                print(f"After lambda net at HLPD iteration {i+1} / {self.hlpd_iterations}, Memory Allocated: {memory_allocated(self.device) / 1e9:.2f} GB")

            # Sigma net

            
            with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                v = self.sigma_nets[i](x, f) 
            if profiling:
                print(f"After sigma net at HLPD iteration {i+1} / {self.hlpd_iterations} , Memory Allocated: {memory_allocated(self.device) / 1e9:.2f} GB")

            # Flow Deformation Operator
            
            if deform_in_fp32:
                with torch.autocast(device_type="cuda", enabled=False):
                    x = self.flow_deform(f[:, [0], ...].float(), v.float()) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).
            else:
                with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                    x = self.flow_deform(f[:, [0], ...], v) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).
            print(f"After flow deformation at HLPD iteration {i+1} /{self.hlpd_iterations}  , Memory Allocated: {memory_allocated(self.device) / 1e9:.2f} GB")
        
        else:
            # Forward projection

            Tf = self.ray_trafo(f)

            # Gamma net

            h = self.gamma_nets[i](h, Tf, y) 

            # Backprojection
            
            TTh = self.ray_trafo.backproject(h)
            

            # Lambda net
            
            f = self.lambda_nets[i](f, TTh, x)

            # Sigma net

            
            v = self.sigma_nets[i](x, f) 

            # Flow Deformation Operator
            
            x = self.flow_deform(f[:, [0], ...] , v) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).


        return f, h, x, v




    def forward(self, y: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:

        self.profiling["memory_allocated"].append(memory_allocated(self.device))
        print(f"Memory Allocated at start of forward pass: {memory_allocated(self.device) / 1e9:.2f} GB")
        """Apply the HLPD model to the input image to produce the deformed image."""
        # Here we would implement the logic for applying the HLPD model to the input image, which typically involves passing the input image through the CNN module to estimate the velocity field, and then applying the deformation module to produce the deformed image.
        
        f, h = self._initialize_hlpd()
        x = torch.zeros_like(f).to(self.device)

        print(f"Memory Allocated after initialization: {memory_allocated(self.device) / 1e9:.2f} GB")


        # AMP logic

        amp_enabled = False
        amp_dtype = torch.bfloat16 if amp_enabled else None
        ray_trafo_in_fp32 = True # NOTE: This is currently the only supported option
        deform_in_fp32 = True

        for i in range(self.hlpd_iterations):
            f, h, x, v = checkpoint(
                lambda f_, h_, y_, x_, i_: self.hlpd_iteration(f_, h_, y_, x_, i_, amp_enabled=amp_enabled, amp_dtype=amp_dtype, ray_trafo_in_fp32=ray_trafo_in_fp32, deform_in_fp32=deform_in_fp32, profiling=True),
                f, h, y, x, i,
                use_reentrant=False
            )
            print(f"After checkpointing at HLPD iteration {i+1} / {self.hlpd_iterations}, Memory Allocated: {memory_allocated(self.device) / 1e9:.2f} GB")

        out ={"x": x, "f": f, "v": v} 
        return out
    

class RecurentHLPD(HLPDModel):

    def __init__(
            self, 
            lambda_channels: list[int],
            gamma_channels: list[int],
            dropout: float = 0.0,
            batch_norm: bool = True,
            hlpd_iterations: int = 10,
            ) -> None:



        super().__init__(
            lambda_channels=lambda_channels,
            gamma_channels=gamma_channels,
            dropout=dropout,
            batch_norm=batch_norm,
            hlpd_iterations=hlpd_iterations
        )




    def _initialize(self):
        self.lambda_nets = [[LambdaBlock(self.T, self.lambda_channels, dropout=self.dropout, batch_norm=self.batch_norm).to(self.device) for _ in range(self.T)] for __ in range(self.hlpd_iterations)] 
        self.gamma_nets = [[GammaBlock(self.T, self.gamma_channels, dropout=self.dropout, batch_norm=self.batch_norm).to(self.device) for _ in range(self.T)] for __ in range(self.hlpd_iterations)]

        # The Sigma nets are shared across all time bins, as they operate on the deformed image and the estimated velocity field, which are not time-dependent. The Sigma nets are initialized as a list of SigmaBlock instances, one for each HLPD iteration.
        self.sigma_nets = [SigmaBlock(self.T, self.gamma_channels, dropout=self.dropout, batch_norm=self.batch_norm).to(self.device) for _ in range(self.hlpd_iterations)]

        # Initialize the operators used in the HLPD model, such as the ray transform and the deformation operator. The operators are typically initialized based on the dimensions of the input image and any other relevant parameters.

        self.ray_trafo, self.flow_deform = self._initialize_operators()

        self.theta = torch.tensor(self.ray_trafo.angles, device=self.device) if self.ray_trafo.angles is not None else None



    def hlpd_iteration(
            self, 
            f: torch.Tensor, 
            h: torch.Tensor, 
            y: torch.Tensor, 
            x: torch.Tensor, 
            i: int, 

        )->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:



        f_new = torch.zeros_like(f)
        h_new = torch.zeros_like(h)

        h_tminus1 = h[:, [-1], ...] 
        f_tminus1 = f[:, [-1], ...]

        for t in range(self.T):

            Tf_t = self.ray_trafo(f[:, [t], ...])
            Tf_tminus1 = self.ray_trafo(f_tminus1)
            
            
            h_t = self.gamma_nets[i][t](torch.cat([h[:, [t], ...], h_tminus1, y[:, [t], ...], Tf_t, Tf_tminus1], dim=1))

            TTh = self.ray_trafo.backproject(h_t)
            TTh_tminus1 = self.ray_trafo.backproject(h_tminus1)

            f_t = self.lambda_nets[i][t](torch.cat([f[:, [t], ...], f_tminus1, x[:, [t], ...], TTh, TTh_tminus1], dim=1))

            # Set the new values of h and f for the current time bin t
        
            h_new[:, [t], ...] = h_t
            f_new[:, [t], ...] = f_t

            # Update the previous values of h and f for the next time bin t+1

            h_tminus1 = h_t
            f_tminus1 = f_t

        # Sigma net

        v = self.sigma_nets[i](x, f) 
        
        # Flow Deformation Operator
        
        x = self.flow_deform(f[:, [0], ...] , v) # Apply deformation to the input image using the estimated velocity field. Here we assume that the first channel of f corresponds to the predicted template, which is used for deformation. This can be customized based on the requirements of the project (e.g., using a different channel of f, etc.).

        return f_new, h_new, x, v