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


def plot_registration_summary_3d(
    template: torch.Tensor,
    target: torch.Tensor,
    velocity_field: torch.Tensor,
    deformed_template: torch.Tensor,
    meta_data: dict,
    savepath: Path = PLOT_DIR / "diffeomorphic_registration_results_3d.png",
) -> Path:
    """3D counterpart of plot_registration_summary: for each of template, target, velocity field magnitude,
    and deformed template (3D volumes of shape (B, D, H, W); velocity_field is (B, D, H, W, 3)), plot the
    axial, coronal, and sagittal mid-slices in a grid and save to disk.

    Slicing and aspect-ratio-correct plotting is delegated to
    visualization.static_visualization.StaticVisualization so the physical geometry in `meta_data`
    (resampled_pixel_spacing / resampled_slice_thickness) is respected, rather than reimplementing plane
    extraction here.
    """

    from visualization.static_visualization import StaticVisualization

    velocity_magnitude = torch.sqrt((velocity_field ** 2).sum(dim=-1))  # (B, D, H, W)

    volumes = {
        "Template": template,
        "Target": target,
        "Velocity Magnitude": velocity_magnitude,
        "Deformed Template": deformed_template,
    }
    planes = ["axial", "coronal", "sagittal"]

    viz = StaticVisualization(meta_data=meta_data, batch_idx=0)
    # All volumes share the same spatial shape, so a single call to _init_shape sets up the geometry
    # (slice indices, physical extent) shared by every plot below.
    viz._init_shape(torch.Size((1, 1, *template.shape[1:])))

    fig, ax = plt.subplots(len(volumes), len(planes), figsize=(4 * len(planes), 4 * len(volumes)))

    for row, (name, vol) in enumerate(volumes.items()):
        vol_5d = vol.detach().cpu().unsqueeze(1)  # (B, 1, D, H, W): add a singleton time-bin axis
        for col, plane in enumerate(planes):
            im, title = viz.visualize_plane(vol_5d, ax[row, col], plane=plane, time_bin=0, prefix=name)
            title.set_fontsize(8)
            if name == "Velocity Magnitude":
                fig.colorbar(im, ax=ax[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()

    savepath = Path(savepath)
    os.makedirs(savepath.parent, exist_ok=True)
    print("Saving 3D diffeomorphic registration results to:", savepath)
    plt.savefig(savepath)
    plt.close(fig)

    return savepath


def plot_deformation_sequence_gif(
    image_sequence: Sequence[torch.Tensor],
    savepath: Path = PLOT_DIR / "deformation_sequence.gif",
    fps: int = 5,
    cmap: str = 'gray',
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Animate a time sequence of deformed images (e.g. from FlowDeformationOperator.forward or
    forward_temporal_superres) as a gif.

    `image_sequence` is a list of tensors, one per time step, each of shape (B, H, W) (2D) or
    (B, D, H, W) (3D, only the middle slice along D is shown). Only the first batch element is plotted.

    `vmin`/`vmax` default to the actual min/max intensity found across the whole sequence rather than
    a fixed [0, 1] range: real volumes (e.g. RegAndReconDataset's `volume` tensor) are not necessarily
    normalized to [0, 1], and hardcoding that range renders everything outside it as flat black/white.
    """

    frames = []
    for img in image_sequence:
        frame = img[0].squeeze().cpu().detach().numpy()
        if frame.ndim == 3:
            # 3D volume: show the middle slice along the first spatial axis.
            frame = frame[frame.shape[0] // 2]
        frames.append(frame)

    if vmin is None:
        vmin = min(frame.min() for frame in frames)
    if vmax is None:
        vmax = max(frame.max() for frame in frames)

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax)
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


def _to_5d_single_timebin(img: torch.Tensor) -> torch.Tensor:
    """Normalize a single deformation-sequence frame to (B, 1, D, H, W) so it can be sliced with
    StaticVisualization, regardless of whether it already carries a channel/time-bin axis: template
    frames from get_3d_target_and_template_from_dataset are (B, D, H, W), while frames produced by
    FlowDeformationOperator (which inserts a channel dim before grid_sample) are (B, 1, D, H, W)."""

    if img.dim() == 4:
        return img.unsqueeze(1)
    elif img.dim() == 5:
        return img
    else:
        raise ValueError(f"Expected a 4D (B, D, H, W) or 5D (B, 1, D, H, W) tensor, but got shape {tuple(img.shape)}.")


def plot_deformation_sequence_gif_3d(
    image_sequence: Sequence[torch.Tensor],
    meta_data: dict,
    savepath: Path = PLOT_DIR / "deformation_sequence_3d.gif",
    fps: int = 5,
    cmap: str = 'gray',
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """3D counterpart of plot_deformation_sequence_gif: animates a time sequence of deformed 3D volumes
    across all major anatomical planes (axial/coronal/sagittal) side by side, one mid-slice per plane
    per frame, instead of a single hardcoded mid-slice.

    `image_sequence` is a list of tensors, one per integration step, each shaped (B, D, H, W) or
    (B, 1, D, H, W) (only the first batch element is plotted). Slicing is delegated to
    visualization.static_visualization.StaticVisualization, the same helper plot_registration_summary_3d
    uses, so both figures agree on what the axial/coronal/sagittal mid-slices are.

    `vmin`/`vmax` default to the actual min/max intensity found across the sequence (see
    plot_deformation_sequence_gif for why a fixed [0, 1] range is wrong for real volumes).
    """

    from visualization.static_visualization import StaticVisualization

    planes = ["axial", "coronal", "sagittal"]

    frames_5d = [_to_5d_single_timebin(img[:1].detach().cpu()) for img in image_sequence]

    viz = StaticVisualization(meta_data=meta_data, batch_idx=0)
    viz._init_shape(frames_5d[0].shape)

    slices_per_frame = [
        [viz._extract_slice(frame, plane=plane, time_bin=0).numpy() for plane in planes]
        for frame in frames_5d
    ]

    if vmin is None:
        vmin = min(s.min() for frame_slices in slices_per_frame for s in frame_slices)
    if vmax is None:
        vmax = max(s.max() for frame_slices in slices_per_frame for s in frame_slices)

    fig, axes = plt.subplots(1, len(planes), figsize=(4 * len(planes), 4.5))
    ims = []
    for col, plane in enumerate(planes):
        im = axes[col].imshow(slices_per_frame[0][col], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[col].set_title(plane.capitalize(), fontsize=10)
        axes[col].axis('off')
        ims.append(im)
    suptitle = fig.suptitle(f"Deformation Sequence (frame 1/{len(frames_5d)})", fontsize=11)

    def update(i):
        for col, im in enumerate(ims):
            im.set_data(slices_per_frame[i][col])
        suptitle.set_text(f"Deformation Sequence (frame {i + 1}/{len(frames_5d)})")
        return ims + [suptitle]

    anim = FuncAnimation(fig, update, frames=len(frames_5d), interval=1000 / fps, blit=False)

    savepath = Path(savepath)
    os.makedirs(savepath.parent, exist_ok=True)
    print("Saving 3D deformation sequence gif to:", savepath)
    anim.save(savepath, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return savepath

# For a 3D deformation field (B, D, H, W, 3), which spatial axis (1=D, 2=H, 3=W) is fixed at a mid-slice
# to obtain each 2D anatomical plane, and which 2 of phi's 3 channels lie within that plane. Channels
# follow grid_sample's (x, y, z) convention -- x=channel 0 aligned with W, y=channel 1 aligned with H,
# z=channel 2 aligned with D (see VelocityIntegrator._identity_grid) -- so e.g. slicing out D (axial)
# leaves the (H, W) plane, whose in-plane displacement is (y, x) = channels (1, 0).
_PLANE_CONFIG = {
    "axial":    {"slice_dim": 1, "row_channel": 1, "col_channel": 0},  # fix D; plane is (H, W)
    "coronal":  {"slice_dim": 2, "row_channel": 2, "col_channel": 0},  # fix H; plane is (D, W)
    "sagittal": {"slice_dim": 3, "row_channel": 2, "col_channel": 1},  # fix W; plane is (D, H)
}


def plot_deformed_grid(
    phi: torch.Tensor,
    grid_size: int = 25,
    plane: str = "axial",
    slice_idx: int | None = None,
    savepath: Path = PLOT_DIR / "deformed_grid.png",
) -> Path:
    """
    Plot a deformed grid based on the given deformation field.

    Parameters:
    - phi: torch.Tensor
        The deformation field of shape (B, H, W, 2) for 2D or (B, D, H, W, 3) for 3D.
    - grid_size: int, optional
        The number of grid lines along each dimension. Default is 10.
    - plane: str, optional
        For a 3D deformation field, which anatomical plane to slice through: "axial", "coronal", or
        "sagittal" (see _PLANE_CONFIG). Ignored for a 2D deformation field. Default is "axial".
    - slice_idx: int, optional
        For a 3D deformation field, the index along the sliced-out axis to use. Defaults to its middle
        index. Ignored for a 2D deformation field.
    - savepath: str, optional
        The path to save the plotted deformed grid. Default is "deformed_grid.png".
    """
    assert phi.dim() in (4, 5) and ((phi.dim() == 4 and phi.shape[-1] == 2) or (phi.dim() == 5 and phi.shape[-1] == 3)), f"Expected a 2D deformation or 3D deformation field of shape (B, H, W, 2) or (B, D, H, W, 3), but got {tuple(phi.shape)}."

    phi = phi.detach().cpu()

    if phi.dim() == 5:
        assert plane in _PLANE_CONFIG, f"Invalid plane {plane!r}. Expected one of {list(_PLANE_CONFIG)}."
        cfg = _PLANE_CONFIG[plane]
        slice_dim = cfg["slice_dim"]
        if slice_idx is None:
            slice_idx = phi.shape[slice_dim] // 2
        phi = phi.select(dim=slice_dim, index=slice_idx)  # (B, *plane_dims, 3)
        row_channel, col_channel = cfg["row_channel"], cfg["col_channel"]
    else:
        row_channel, col_channel = 1, 0  # (B, H, W, 2): row axis H uses channel 1 (y), col axis W uses channel 0 (x)

    # phi is normalized to [-1, 1] (grid_sample's convention). Undo that back to pixel coordinates,
    # scaling each axis by its OWN size -- critical for a non-cubic plane (e.g. coronal/sagittal's D axis
    # is typically much smaller than H/W), where using the wrong axis's size silently compresses that
    # dimension instead of raising an error.
    row_size, col_size = phi.shape[1], phi.shape[2]
    phi_px = torch.empty(*phi.shape[:3], 2)
    phi_px[..., 0] = (phi[..., col_channel] + 1) / 2 * (col_size - 1)
    phi_px[..., 1] = (phi[..., row_channel] + 1) / 2 * (row_size - 1)

    # Evenly spaced row/column indices across the full extent, not just the first grid_size entries.
    row_idx = torch.linspace(0, row_size - 1, grid_size).round().long()
    col_idx = torch.linspace(0, col_size - 1, grid_size).round().long()

    fig, ax = plt.subplots()
    for i in row_idx:
        ax.plot(phi_px[0, i, :, 0], phi_px[0, i, :, 1], 'r', linewidth=0.8)
    for j in col_idx:
        ax.plot(phi_px[0, :, j, 0], phi_px[0, :, j, 1], 'r', linewidth=0.8)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # match imshow's default origin='upper' (row axis increases downward)

    savepath = Path(savepath)
    os.makedirs(savepath.parent, exist_ok=True)
    print(f"Saving deformed grid to: {savepath}")
    plt.savefig(savepath)
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
