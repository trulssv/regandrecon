import torch
import torch.nn as nn
import torch.nn.functional as F


class RegistrationCNN(nn.Module):
    """This is a simple registration network that takes a 2D template and target and outputs a velocity field that can be used to deform the template to match the target."""

    def __init__(self, input_channels=2, hidden_channels=32, output_channels=2):
        super(RegistrationCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, template, target):
        x = torch.stack((template, target), dim=1)  # Concatenate along the channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        velocity_field = self.conv3(x)  # 1, 2, H, W

        velocity_field = velocity_field.permute(0, 2, 3, 1)  # Rearrange to (B, H, W, 2)

        return velocity_field


def _num_groups(num_channels, max_groups=8):
    """Largest group count <= max_groups that evenly divides num_channels (falls back to 1)."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by group normalization and a ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = _num_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling block: max pool followed by a double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upscaling block: bilinear upsample, concatenate with the skip connection, then a double convolution.

    The skip connection may be a slightly different spatial size than the upsampled feature map when the
    input resolution is not divisible by 2**depth, so the upsampled map is resized to match before concatenation.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """A UNet architecture for estimating a velocity field from a 2D template and target image.

    Compared to RegistrationCNN, the encoder-decoder structure with skip connections gives the network a much
    larger receptive field and capacity, which matters when the resolution (and thus the deformation) increases.
    """

    def __init__(self, in_channels=2, out_channels=2, base_channels=32, depth=4):
        super().__init__()
        self.depth = depth

        self.inc = DoubleConv(in_channels, base_channels)

        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.downs = nn.ModuleList([Down(channels[i], channels[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList(
            [Up(channels[i + 1], channels[i], channels[i]) for i in reversed(range(depth))]
        )

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, template, target):
        x = torch.stack((template, target), dim=1)  # (B, 2, H, W)

        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        velocity_field = self.outc(x)  # (B, 2, H, W)
        velocity_field = velocity_field.permute(0, 2, 3, 1)  # (B, H, W, 2)

        return velocity_field
