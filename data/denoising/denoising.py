import deepinv as dinv
import torch

class Denoiser(torch.nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.model = self.load_denoiser(device=device)


    def load_denoiser(self, device="cuda:0"):

        PATH_TO_DENOISER = "data/denoising/model/drunet_deepinv_gray.pth"

        # Load denoiser

        model = dinv.models.DRUNet(
            in_channels=1, 
            out_channels=1, 
            pretrained=PATH_TO_DENOISER,
            pretrained_2d_isotropic=True,
            device=device,
            dim=2
            )
        
        return model

    def forward(self, x, sigma):
        """This function applies the denoiser to the input volume x. The input volume x is expected to have the shape (B, T, D, H, W) where B is the batch size, T is the number of time bins, D is the depth of the volume, H is the height of the volume and W is the width of the volume. The output of this function will have the same shape as the input volume x."""
        B, T, D, H, W = x.shape        

        # iterate over all time bins:

        for t in range(T):

            print(f"Denoising time bin {t+1}/{T}...")

            x_t = x[:, t]  # Shape: (B, D, H, W)

            # convert to 2D slices and apply denoiser to each slice

            x_t = x_t.view(B*D, 1, H, W)  # Shape: (B*D, 1, H, W)

            with torch.no_grad():
                x_denoised = self.model(x_t, sigma=sigma)
    
            x[:, t] = x_denoised.view(B, D, H, W)


        return x


def main():

    model = Denoiser()


if __name__ == "__main__":
    main()