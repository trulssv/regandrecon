import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """This class implements a simple convolutional block, which consists of a convolutional layer followed by a ReLU activation function. 
    This block can be used as a building block for the CNN modules in the HLPD model."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True, final: bool = False)->None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.activation = nn.LeakyReLU(negative_slope=0.01) if not final else nn.Identity() # We use LeakyReLU as the activation function for all blocks except the final block, where we use an identity function to allow for unbounded output values. This can be customized based on the requirements of the project (e.g., using different activation functions, etc.).
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 and not final else nn.Identity()
        # self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=out_channels) if batch_norm and not final else nn.Identity() # GroupNorm can be used as an alternative to BatchNorm, especially for small batch sizes. Here we use 8 groups, but this can be customized based on the requirements of the project (e.g., using different numbers of groups, etc.).

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """Apply the convolutional block to the input tensor."""

        
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.group_norm(x)
        return x




class CNNModule(nn.Module):
    """This class implements a simple CNN module, which consists of a sequence of convolutional blocks with specified channels, kernel size, padding, stride, dropout, and batch normalization. 
    This module can be used as a building block for the HLPD model, and can be customized based on the requirements of the project (e.g., using different numbers of convolutional blocks, etc.)."""
    def __init__(self, channels: list[int], kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True)->None:
        assert len(channels) >= 2, "Channels list must have at least 2 elements (input and output channels)."
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], kernel_size, padding, stride, dropout, batch_norm, final=True if i == len(channels) - 2 else False)) # We set final=True for the last block to allow for unbounded output values, which can be useful for regression tasks. This can be customized based on the requirements of the project (e.g., using different activation functions for the final block, etc.).

    def forward(self, *args) -> torch.Tensor:
        """Apply the CNN module to the input tensor."""
        assert all(isinstance(arg, torch.Tensor) for arg in args), "All inputs must be torch.Tensors. Received inputs of types: " + ", ".join(type(arg).__name__ for arg in args)

        x = torch.cat(args, dim=1) if len(args) > 1 else args[0]
        for block in self.blocks:
            x = block(x)
        return x


class ResidualCNNModule(CNNModule):
    """This class implements a residual CNN module, which consists of a sequence of residual convolutional blocks with specified channels, kernel size, padding, stride, dropout, and batch normalization. 
    This module can be used as a building block for the HLPD model, and can help to improve the convergence and stability of the model during training."""

    def __init__(self, channels: list[int], kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True, tau=None)->None:
        super().__init__(channels, kernel_size, padding, stride, dropout, batch_norm)
        self.tau = tau if tau is not None else 1.0 # The tau parameter can be used to control the strength of the residual connection, which can help to improve the convergence and stability of the model during training. A value of tau=1.0 corresponds to a standard residual connection, while values less than 1.0 can be used to weaken the residual connection and encourage the model to learn more from the convolutional blocks. This can be customized based on the requirements of the project (e.g., using different values of tau, etc.).


    def forward(self, *args) -> torch.Tensor:
        """Apply the residual CNN module to the input tensor."""
        assert all(isinstance(arg, torch.Tensor) for arg in args), "All inputs must be torch.Tensors. Received inputs of types: " + ", ".join(type(arg).__name__ for arg in args)
        skip = args[0] 
        x = torch.cat(args, dim=1) if len(args) > 1 else args[0]


        for block in self.blocks:
            x = block(x)

        assert skip.shape == x.shape, f"Skip connection shape {skip.shape} does not match input shape {x.shape}. Please check the input format and the architecture of the CNN module to ensure that the skip connection is properly aligned with the input tensor."

        return skip + self.tau * x


class FiLMLayer(nn.Module):
    """This class implements a FiLM layer, which encodes the source angle as conditioning information to modulate the features of the input tensor. 
    The FiLMLayer can be implemented as a simple feedforward neural network that takes the conditioning information (e.g., source angle) as input and produces scaling (gamma) and shifting (beta) parameters for the input tensor.
    """
    
    def __init__(self, in_channels: int, out_channels: int, d_emb: int)->None:
        super().__init__()
        self.d_emb = d_emb # The dimensionality of the embedding for the conditioning information. This can be customized based on the requirements of the project (e.g., using different embedding sizes, etc.).
        
        assert d_emb % 2 == 0, "Embedding dimension d_emb must be even."

        # NOTE: in this case in_channels = 3 * T and out_channels = T, so the order is reversed in the linear layers

        self.gamma = nn.Linear(d_emb * out_channels, in_channels) # The gamma parameter is used to scale the features of the input tensor based on the conditioning information.
        self.beta = nn.Linear(d_emb * out_channels, in_channels) # The beta parameter is used to shift the features of the input tensor based on the conditioning information.

    def encode(self, theta: torch.Tensor) -> torch.Tensor:
        """Encode the conditioning information (e.g., source angle) into a feature representation that can be used to modulate the input tensor."""
        
        
        B, T = theta.shape

        x = torch.zeros(B, T, self.d_emb, device=theta.device) # Initialize the embedding tensor with zeros.

        # Encode theta_i, i=1,.., T as a d_emb-dimensioal embedding using sine and cosine functions, similar to the positional encoding used in transformers. This allows the model to capture the periodic nature of the source angle and its relationship to the features of the input tensor.
        for i in range(T):
            for j in range(self.d_emb // 2):
                x[:, i, 2*j] = torch.sin(theta[:, i] * j) # Sine encoding for the even dimensions of the embedding.
                x[:, i, 2*j + 1] = torch.cos(theta[:, i] * j) # Cosine encoding for the odd dimensions of the embedding.

        return x # In this simple implementation, we directly use the input theta as the conditioning information. This can be customized based on the requirements of the project (e.g., using a more complex encoding of the conditioning information, etc.).

    def forward(self, x: torch.Tensor, theta: torch.Tensor)->torch.Tensor:
        """Apply feature-wise linear modulation to the input tensor based on the conditioning information."""
        emb = self.encode(theta)
        emb = emb.view(emb.size(0), -1) # Flatten the embedding tensor to match the input format of the linear layers for gamma and beta.
        gamma = self.gamma(emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Reshape gamma to match the dimensions of x for broadcasting.
        beta = self.beta(emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Reshape beta to match the dimensions of x for broadcasting.
        return gamma * x + beta

class FiLMCNNModule(CNNModule):
    """This class implements a FiLM CNN module, which consists of a sequence of convolutional blocks with FiLM modulation based on the conditioning information (e.g., source angle). 
    This module can be used as a building block for the HLPD model, and can help to improve the performance of the model by allowing it to adapt its features based on the conditioning information."""
    
    def __init__(self, channels: list[int], d_emb: int=20, kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True)->None:
        super().__init__(channels, kernel_size, padding, stride, dropout, batch_norm)
        
        self.film_layer = FiLMLayer(channels[0], channels[-1], d_emb) # The FiLMLayer is used to modulate the features of the input tensor based on the conditioning information (e.g., source angle).

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Apply the FiLM CNN module to the input tensor with FiLM modulation based on the conditioning information."""
        x = self.film_layer(x, theta) # Apply FiLM modulation to the input tensor based on the conditioning information.
        for block in self.blocks:
            x = block(x)
        return x

class ResidualFiLMCNNModule(FiLMCNNModule):
    """This class implements a residual FiLM CNN module, which consists of a sequence of residual convolutional blocks with FiLM modulation based on the conditioning information (e.g., source angle). 
    This module can be used as a building block for the HLPD model, and can help to improve the performance of the model by allowing it to adapt its features based on the conditioning information while also benefiting from the advantages of residual connections (e.g., improved convergence and stability during training)."""
    def __init__(self, channels: list[int], d_emb: int=20, kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True, tau=None)->None:
        super().__init__(channels, d_emb, kernel_size, padding, stride, dropout, batch_norm)
        self.tau = tau if tau is not None else 1.0 # The tau parameter can be used to control the strength of the residual connection, which can help to improve the convergence and stability of the model during training. A value of tau=1.0 corresponds to a standard residual connection, while values less than 1.0 can be used to weaken the residual connection and encourage the model to learn more from the convolutional blocks. This can be customized based on the requirements of the project (e.g., using different values of tau, etc.).

    def forward(self, *args) -> torch.Tensor:
        """Apply the residual FiLM CNN module to the input tensor with FiLM modulation based on the conditioning information."""

        assert all(isinstance(arg, torch.Tensor) for arg in args), "All inputs must be torch.Tensors. Received inputs of types: " + ", ".join(type(arg).__name__ for arg in args)
        assert len(args) >= 2, "ResidualFiLMCNNModule requires at least 2 input tensors: the skip connection tensor and the conditioning information tensor (e.g., source angle). Received " + str(len(args)) + " input tensors."

        skip = args[0]
        theta = args[-1]
        x = torch.cat(args[:-1], dim=1) if len(args) > 2 else args[0] # Concatenate all inputs except the last one (which is theta) along the channel dimension to form the input tensor for the convolutional blocks.
        x = self.film_layer(x, theta) # Apply FiLM modulation to the input tensor based on the conditioning information.
        for block in self.blocks:
            x = block(x)

        assert skip.shape == x.shape, f"Skip connection shape {skip.shape} does not match input shape {x.shape}. Please check the input format and the architecture of the CNN module to ensure that the skip connection is properly aligned with the input tensor."

        return skip + self.tau * x

class GammaBlock(CNNModule):
    """This is an implementation of the gamma block:
    
    h_new = GammaBlock(h, Tf, y)

    where h is the current estimate of the sinogram, Tf is the forward projection of the current estimate of the image, and y is the measured sinogram. 
    The GammaBlock can be implemented as a simple feedforward neural network that takes the concatenation of h, Tf, and y as input and produces an updated estimate of the sinogram h_new as output.     
    """

    def __init__(self, T: int, hidden_channels: list[int], kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True)->None:
        in_channels = 3 * T # We concatenate h, Tf, and y along the channel dimension, so the input channels will be 3 times the number of time bins T.
        out_channels = T # The output channels will be equal to the number of time bins T
        channels = [in_channels] + hidden_channels + [out_channels]
        super().__init__(channels, kernel_size, padding, stride, dropout, batch_norm)
            
class ResidualGammaBlock(ResidualCNNModule):
    """This is an implementation of a residual gamma block, which consists of a gamma block followed by a skip connection that adds the input to the output of the gamma block. 
    This block can be used as a building block for the HLPD model, and can help to improve the convergence and stability of the model during training."""
    def __init__(self, T: int, hidden_channels: list[int], kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True, tau=None)->None:
        in_channels = 3 * T # We concatenate h, Tf, and y along the channel dimension, so the input channels will be 3 times the number of time bins T.
        out_channels = T # The output channels will be equal to the number of time bins T
        channels = [in_channels] + hidden_channels + [out_channels]
        super().__init__(channels, kernel_size, padding, stride, dropout, batch_norm, tau)

class ResidualFiLMGammaBlock(ResidualFiLMCNNModule):
    """This is an implementation of a residual FiLM gamma block, which consists of a gamma block with FiLM modulation based on the conditioning information (e.g., source angle) followed by a skip connection that adds the input to the output of the gamma block. 
    This block can be used as a building block for the HLPD model, and can help to improve the performance of the model by allowing it to adapt its features based on the conditioning information while also benefiting from the advantages of residual connections (e.g., improved convergence and stability during training)."""
    def __init__(
            self,
            T: int, 
            hidden_channels: list[int]=[64, 64], 
            kernel_size: int = 3, 
            padding: int = 1, 
            stride: int = 1, 
            dropout: float = 0.0, 
            batch_norm: bool = True, 
            tau=None,
            d_emb: int = 8
            )->None:
        in_channels = 3 * T # We concatenate h, Tf, and y along the channel dimension, so the input channels will be 3 times the number of time bins T.
        out_channels = T # The output channels will be equal to the number of time bins T
        channels = [in_channels] + hidden_channels + [out_channels]
        super().__init__(channels, d_emb, kernel_size, padding, stride, dropout, batch_norm, tau)

class ResidualLambdaBlock(ResidualCNNModule):
    """This is an implementation of a residual lambda block, which consists of a lambda block followed by a skip connection that adds the input to the output of the lambda block. 
    This block can be used as a building block for the HLPD model, and can help to improve the convergence and stability of the model during training."""
    def __init__(self, T: int, hidden_channels: list[int], kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True, tau=None)->None:
        in_channels = 3 * T # We concatenate f, TTh, and x along the channel dimension, so the input channels will be 3 times the number of time bins T.
        out_channels = T # The output channels will be equal to the number of time bins T
        channels = [in_channels] + hidden_channels + [out_channels]
        super().__init__(channels, kernel_size, padding, stride, dropout, batch_norm, tau)

class LambdaBlock(CNNModule):
    """This is an implementation of the lambda block:
    
    f_new = LambdaBlock(f, TTh, x)

    where f is the current estimate of the image, TTh is the backprojection of the current estimate of the sinogram, and x the registered image. 
    The LambdaBlock can be implemented as a simple feedforward neural network that takes the concatenation of f, TTh, and x as input and produces an updated estimate of the image f_new as output.     
    """

    def __init__(self, T: int, hidden_channels: list[int], kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True)->None:
        in_channels = 3 * T # We concatenate f, TTh, and x along the channel dimension, so the input channels will be 3 times the number of time bins T.
        out_channels = T # The output channels will be equal to the number of time bins T
        channels = [in_channels] + hidden_channels + [out_channels]
        super().__init__(channels, kernel_size, padding, stride, dropout, batch_norm)
        


class SigmaBlock(CNNModule):
    """
    This is an implementation of the sigma block:
    
    v_new = SigmaBlock(x, f)

    where x is the current registration, and f is the reconstruction. 
    The SigmaBlock can be implemented as a simple feedforward neural network that takes the concatenation of x and f as input and produces an updated estimate of the velocity field v_new as output.     
    """

    def __init__(self, T: int, hidden_channels: list[int], kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.0, batch_norm: bool = True)->None:
        in_channels = 2 * T # We concatenate x and f along the channel dimension, so the input channels will be 2 times the number of time bins T.

        # NOTE: The output channels of the sigma block should be 3 times the number of time bins T-1, since we need to produce a velocity field with 3 components (vx, vy, vz) for each time bin. The number of time bins is T-1 
        # because we are estimating the velocity field between consecutive time bins, so we have one less velocity field than the number of time bins.

        out_channels = 3 * (T-1) # The output channels will be equal to 3 times the number of time bins T-1, since we need to produce a velocity field with 3 components (vx, vy, vz) for each time bin.
        channels = [in_channels] + hidden_channels + [out_channels]
        super().__init__(channels, kernel_size, padding, stride, dropout, batch_norm)



    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Apply the Sigma block to the input tensor."""
        x = torch.cat((x, f), dim=1)
        for block in self.blocks:
            x = block(x)
        
        # NOTE: output has shape B x 3(T-1) x D x H x W, we need to reshape it to B x (T-1) x D x H x W x 3, where the 3 corresponds to the velocity components (vx, vy, vz) for each time bin.
        B, C, D, H, W = x.shape
        t = C // 3 # T-1
        x = x.view(B, t, 3, D, H, W)      # keep channel grouping explicit
        x = x.permute(0, 1, 3, 4, 5, 2)   # -> (B, t, D, H, W, 3)
        return x




    
