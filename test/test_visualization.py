import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

PLOT_DIR = Path("test/plots")


def plot_registration_summary(
    template: torch.Tensor,
    target: torch.Tensor,
    velocity_field: torch.Tensor,
    deformed_template: torch.Tensor,
    savepath: Path = PLOT_DIR / "diffeomorphic_registration_results.png",
) -> Path:
    """Plot the template, target, velocity field magnitude, and deformed template in a 2x2 grid and save to disk."""

    fig, ax = plt.subplots(2, 2)

    im = ax[0, 0].imshow(template.squeeze().cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title('Template Image')
    ax[0, 0].axis('off')

    im = ax[0, 1].imshow(target.squeeze().cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title('Target Image')
    ax[0, 1].axis('off')

    im = ax[1, 0].imshow(torch.sqrt(velocity_field[0, ..., 0] ** 2 + velocity_field[0, ..., 1] ** 2).cpu().detach().numpy(), cmap='hot')
    ax[1, 0].set_title('Velocity Field Magnitude')
    ax[1, 0].axis('off')
    fig.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(deformed_template.squeeze().cpu().detach().numpy(), cmap='gray')
    ax[1, 1].set_title('Deformed Template')
    ax[1, 1].axis('off')
    fig.colorbar(im, ax=ax[1, 1])

    plt.tight_layout()

    savepath = Path(savepath)
    os.makedirs(savepath.parent, exist_ok=True)
    print("Saving diffeomorphic registration results to:", savepath)
    plt.savefig(savepath)
    plt.close(fig)

    return savepath


def plot_deformation_sequence_gif(
    image_sequence: Sequence[torch.Tensor],
    savepath: Path = PLOT_DIR / "deformation_sequence.gif",
    fps: int = 5,
    cmap: str = 'gray',
) -> Path:
    """Animate a time sequence of deformed images (e.g. from FlowDeformationOperator.forward or
    forward_temporal_superres) as a gif.

    `image_sequence` is a list of tensors, one per time step, each of shape (B, H, W) (2D) or
    (B, D, H, W) (3D, only the middle slice along D is shown). Only the first batch element is plotted.
    """

    frames = []
    for img in image_sequence:
        frame = img[0].squeeze().cpu().detach().numpy()
        if frame.ndim == 3:
            # 3D volume: show the middle slice along the first spatial axis.
            frame = frame[frame.shape[0] // 2]
        frames.append(frame)

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=1)
    title = ax.set_title(f"Deformation Sequence (frame 1/{len(frames)})")
    ax.axis('off')

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"Deformation Sequence (frame {i + 1}/{len(frames)})")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    savepath = Path(savepath)
    os.makedirs(savepath.parent, exist_ok=True)
    print("Saving deformation sequence gif to:", savepath)
    anim.save(savepath, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return savepath


def main():
    from test.test_diffeomorphic_registration import diffeomorphic_registration, get_target_and_template, setup_operators
    from test.test_models import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    shape = (128, 128)
    template, target = get_target_and_template(device=device, shape=shape)

    velocity_field, deformed_template = diffeomorphic_registration(
        template, target,
        num_epochs=500, learning_rate=1e-2, lambda_reg=1e-6, device=device,
        model_cls=UNet, model_kwargs={"base_channels": 32, "depth": 4},
    )

    plot_registration_summary(template, target, velocity_field, deformed_template)

    # Integrate the estimated velocity field at its native temporal resolution (flow_deform_op.N)
    # and animate the resulting sequence of intermediate deformations. Requesting N much larger than
    # this underflows float32 precision in the scaling-and-squaring integration (v is scaled by 2**N
    # before being squared back up), so we don't ask for a finer resolution than the operator supports.
    flow_deform_op, _ = setup_operators(device=device)
    deformation_sequence = flow_deform_op.forward_temporal_superres(template, velocity_field, N=flow_deform_op.N)

    plot_deformation_sequence_gif(deformation_sequence)


if __name__ == "__main__":
    main()
