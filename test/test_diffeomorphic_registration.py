from pathlib import Path

import torch
from operators.lddmm.lddmm_loss import LDDMMloss
from tqdm import tqdm

from test.test_models import RegistrationCNN, UNet, RegistrationCNN3D, UNet3D
from test.test_visualization import plot_deformation_sequence_gif_3d, plot_registration_summary_3d, plot_deformed_grid, plot_deformation_sequence_gif, plot_registration_summary

from data.data_loaders import RegAndReconDataset

HELMHOLTZ_PARAMS: dict = {
    "dim": 2,
    "alpha": 1.0,
    "gamma": 1.0,
    "beta": 1.0,
    "extent": ((0.0, 450.0), (0.0, 450.0)),
    "return_fft": False,
}
DEFORM_PARAMS: dict = {
     "N": 7,  # Default number of integration steps for the velocity field
    "extent": ((0.0, 450.0), (0.0, 450.0)),
    "action": "geometric",
    "integration": "euler",
}

# 3D counterparts of HELMHOLTZ_PARAMS/DEFORM_PARAMS above. `extent` is deliberately omitted here since,
# unlike the synthetic 2D extent, the physical extent of a real 3D volume depends on its meta_data
# (resampled_pixel_spacing / resampled_slice_thickness) and is computed per-study by compute_3d_extent().
HELMHOLTZ_PARAMS_3D: dict = {
    "dim": 3,
    "alpha": 1.0,
    "gamma": 1.0,
    "beta": 1.0,
    "return_fft": False,
}
DEFORM_PARAMS_3D: dict = {
    "N": 7,  # Default number of integration steps for the velocity field
    "action": "geometric",
    "integration": "euler",
}


def get_target_and_template(device: torch.device, shape=(128, 128)):
        """This function generates a simple target and template image for testing purposes. The template is a circle, and the target is a square. Both template and target are binary 2D images of size 128x128."""

        template = torch.zeros((1, *shape), dtype=torch.float32, device=device)
        target = torch.zeros((1, *shape), dtype=torch.float32, device=device)

        # Create a square in the target

        Xquarter = shape[0] // 4
        Yquarter = shape[1] // 4
        target[:, Xquarter:3*Xquarter, Yquarter:3*Yquarter] = 1.0

        # Create a circle in the template
        Y, X = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
        center = shape[0] // 2
        radius = shape[0] // 4
        mask = (X - center) ** 2 + (Y - center) ** 2 <= radius ** 2
        template[:, mask] = 1.0

        return template, target

def get_target_and_template_from_dataset(idx: int| None = None, device: torch.device=torch.device("cpu")):



    dataset = RegAndReconDataset(qualities=["high"], mode="val")
    if idx is None:
        idx = 0

    
    data = dataset[idx]

    print(data.keys())

    template = data["volume"].to(device)
    target = data["volume_processed"].to(device)
    return template, target


def get_3d_target_and_template_from_dataset(
    idx: int | None = None,
    time_bins: tuple[int, int] = (0, 4),
    device: torch.device = torch.device("cpu"),
):
    """Extract two 3D volumes (D, H, W) from the same 4D spatio-temporal (T, D, H, W) study, using the
    given pair of time bins (default: the 1st and 5th, i.e. indices 0 and 4) as template and target for
    the 3D diffeomorphic registration pipeline. Also returns the study's meta_data, which the 3D pipeline
    needs to compute a physically correct extent for the Helmholtz regularizer/deformation operator."""

    dataset = RegAndReconDataset(qualities=["high"], mode="val")
    if idx is None:
        idx = 0

    data = dataset[idx]

    volume = data["volume"]  # (T, D, H, W)
    t_template, t_target = time_bins
    template = volume[t_template].unsqueeze(0).to(device)  # (1, D, H, W)
    target = volume[t_target].unsqueeze(0).to(device)  # (1, D, H, W)
    meta_data = data["meta_data"]

    return template, target, meta_data


def compute_3d_extent(
    meta_data: dict, shape: tuple[int, int, int]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute the physical extent (in mm) of a (D, H, W) volume from its meta_data, for use as the
    `extent` kwarg of HelmholtzOperator(dim=3, ...) / FlowDeformationOperator(..., extent=...). The
    returned tuple is ordered (D, H, W) to match the spatial axis order of the (B, D, H, W, 3) velocity
    field, the same convention used by visualization/static_visualization.py's StaticVisualization."""

    d, h, w = shape
    pixel_spacing = meta_data["resampled_pixel_spacing"]  # (W spacing, H spacing)
    slice_thickness = meta_data["resampled_slice_thickness"]

    Z = d * slice_thickness
    Y = h * pixel_spacing[1]
    X = w * pixel_spacing[0]

    return ((0.0, float(Z)), (0.0, float(Y)), (0.0, float(X)))


def setup_operators(device: torch.device):
    from operators.lddmm.deform import FlowDeformationOperator
    from operators.lddmm.helmholtz import HelmholtzOperator

    flow_deform_op = FlowDeformationOperator(**DEFORM_PARAMS, device=device)
    helmholtz_op = HelmholtzOperator(**HELMHOLTZ_PARAMS, device=device)

    return flow_deform_op, helmholtz_op


def setup_operators_3d(shape: tuple[int, int, int], extent: tuple[tuple[float, float], ...], device: torch.device):
    from operators.lddmm.deform import FlowDeformationOperator
    from operators.lddmm.helmholtz import HelmholtzOperator

    flow_deform_op = FlowDeformationOperator(**DEFORM_PARAMS_3D, shape=shape, extent=extent, device=device)
    helmholtz_op = HelmholtzOperator(**HELMHOLTZ_PARAMS_3D, extent=extent, device=device)

    return flow_deform_op, helmholtz_op

def diffeomorphic_registration(template, target, num_epochs=100, learning_rate=1e-3, lambda_reg=0.1, device=torch.device("cpu"), model_cls:type[RegistrationCNN | UNet]=RegistrationCNN, model_kwargs=None):
    """This function performs diffeomorphic registration of the template to the target using a registration network and the LDDMM loss function.
    The registration is performed over a specified number of epochs with a given learning rate.

    `model_cls` selects the network used to estimate the velocity field (e.g. RegistrationCNN or UNet, both defined
    in test/test_models.py), and `model_kwargs` are passed through to its constructor."""

    # Initialize the registration network and loss function
    reg_net = model_cls(**(model_kwargs or {})).to(device)

    flow_deform_op, helmholtz_op = setup_operators(device=device)
    lddmm_loss = LDDMMloss(flow_deform_op=flow_deform_op, helmholtz_op=helmholtz_op, lambda_reg=lambda_reg, device=device)

    # Use Adam optimizer for training
    optimizer = torch.optim.Adam(reg_net.parameters(), lr=learning_rate)

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in pbar:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass: compute the velocity field from the template and target
        velocity_field = reg_net(target, template)

        # Compute the LDDMM loss
        loss, deformed_template = lddmm_loss(template, target, velocity_field)

        # Backward pass: compute gradients
        loss.total_loss.backward()

        # Update the network parameters
        optimizer.step()

        # Update the progress bar and print the current loss values

        pbar.set_postfix({
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Total Loss": f"{loss.total_loss.item():.4f}",
            "Similarity Loss": f"{loss.data_loss.item():.4f}",
            "Regularization Loss": f"{loss.regularization_loss.item():.4f}"
        })



    return velocity_field, deformed_template


def diffeomorphic_registration_3d(
    template, target, meta_data,
    num_epochs=100, learning_rate=1e-3, lambda_reg=0.1,
    device=torch.device("cpu"),
    model_cls: type[RegistrationCNN3D | UNet3D] = RegistrationCNN3D,
    model_kwargs=None,
):
    """3D counterpart of diffeomorphic_registration: performs diffeomorphic registration between two 3D
    volumes (`template`, `target`, both shaped (B, D, H, W), e.g. two time bins of the same spatio-temporal
    study) using a Conv3d velocity-estimation network (RegistrationCNN3D or UNet3D, both defined in
    test/test_models.py) and the LDDMM loss. The physical extent used by the Helmholtz regularizer and the
    deformation operator is derived from `meta_data` (resampled_pixel_spacing / resampled_slice_thickness),
    so the registration is physically accurate for the real dataset volumes rather than a fixed placeholder.
    """

    shape = tuple(template.shape[1:])  # (D, H, W)
    extent = compute_3d_extent(meta_data, shape)

    # Initialize the registration network and loss function
    reg_net = model_cls(**(model_kwargs or {})).to(device)

    flow_deform_op, helmholtz_op = setup_operators_3d(shape=shape, extent=extent, device=device)
    lddmm_loss = LDDMMloss(flow_deform_op=flow_deform_op, helmholtz_op=helmholtz_op, lambda_reg=lambda_reg, device=device)

    optimizer = torch.optim.Adam(reg_net.parameters(), lr=learning_rate)

    pbar = tqdm(range(num_epochs), desc="3D Training Progress")

    for epoch in pbar:
        optimizer.zero_grad()

        velocity_field = reg_net(target, template)

        loss, deformed_template = lddmm_loss(template, target, velocity_field)

        loss.total_loss.backward()

        optimizer.step()

        pbar.set_postfix({
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Total Loss": f"{loss.total_loss.item():.4f}",
            "Similarity Loss": f"{loss.data_loss.item():.4f}",
            "Regularization Loss": f"{loss.regularization_loss.item():.4f}"
        })

    # Returning flow_deform_op lets the caller re-integrate/animate the learned velocity field without
    # re-deriving the same data-dependent extent used during training.
    return velocity_field, deformed_template, flow_deform_op


def main():
    # Generate the target and template images
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sz = 128
    shape = (sz, sz)

    template, target = get_target_and_template(device=device, shape=shape)



    # Perform diffeomorphic registration.
    # RegistrationCNN is a small, fixed-receptive-field network that tends to fail as resolution increases.
    # UNet has a much larger receptive field/capacity via its encoder-decoder structure and skip connections,
    # so it copes better at higher resolutions. Swap model_cls/model_kwargs below to try either.
    velocity_field, deformed_template = diffeomorphic_registration(
        template, target,
        num_epochs=1000, learning_rate=1e-2, lambda_reg=1e-8, device=device,
        model_cls=UNet, model_kwargs={"base_channels": 32, "depth": 4},
    )


    plot_registration_summary(template, target, velocity_field, deformed_template)

    # Integrate the estimated velocity field at its native temporal resolution (flow_deform_op.N)
    # and animate the resulting sequence of intermediate deformations. Requesting N much larger than
    # this underflows float32 precision in the scaling-and-squaring integration (v is scaled by 2**N
    # before being squared back up), so we don't ask for a finer resolution than the operator supports.
    flow_deform_op, _ = setup_operators(device=device)
    deformation_sequence = flow_deform_op.forward(template, velocity_field, superres=True)

    # Add template in the begining of the deformation sequence
    deformation_sequence = [template] + deformation_sequence

    plot_deformation_sequence_gif(deformation_sequence)

    # flow_deform_op.forward()/the training loop always normalize v to grid_sample's [-1, 1] range
    # before integrating (see FlowDeformationOperator.forward). velocity_integrator itself does not do
    # this, so it must be applied explicitly here too -- skipping it feeds raw pixel-scale displacements
    # into the integrator, which produces a garbled, non-continuous phi.
    phi = flow_deform_op.velocity_integrator(flow_deform_op._normalize(velocity_field))
    print(phi.shape)
    plot_deformed_grid(phi)
    


def main_3d():
    """3D counterpart of main(): registers the 1st and 5th time bins of a real 4D spatio-temporal study
    against each other using a 3D velocity-estimation network, then visualizes the result across all major
    anatomical planes (axial/coronal/sagittal)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    template, target, meta_data = get_3d_target_and_template_from_dataset(idx=None, time_bins=(0, 4), device=device)

    # base_channels/depth are kept modest relative to the 2D UNet default (32/4) since a full-resolution
    # (D, H, W) volume is far more memory-hungry per channel than a 2D slice; increase if GPU memory allows.
    velocity_field, deformed_template, flow_deform_op = diffeomorphic_registration_3d(
        template, target, meta_data,
        num_epochs=200, learning_rate=1e-2, lambda_reg=1e-8, device=device,
        model_cls=UNet3D, model_kwargs={"base_channels": 16, "depth": 3},
    )

    plot_registration_summary_3d(template, target, velocity_field, deformed_template, meta_data)

    deformation_sequence = flow_deform_op.forward(template, velocity_field, superres=True)
    deformation_sequence = [template] + deformation_sequence

    plot_deformation_sequence_gif_3d(deformation_sequence, meta_data, savepath=Path("test/plots/deformation_sequence_3d.gif"))

    phi = flow_deform_op.velocity_integrator(flow_deform_op._normalize(velocity_field))
    print(phi.shape)
    plot_deformed_grid(phi)


if __name__ == "__main__":

    d = 3
    if d == 2:
        main()
    elif d == 3:
        main_3d()

