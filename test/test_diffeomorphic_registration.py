import torch
from operators.lddmm.lddmm_loss import LDDMMloss
from tqdm import tqdm
from test.test_models import RegistrationCNN, UNet
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


def setup_operators(device: torch.device):
    from operators.lddmm.deform import FlowDeformationOperator
    from operators.lddmm.helmholtz import HelmholtzOperator

    flow_deform_op = FlowDeformationOperator(**DEFORM_PARAMS, device=device)
    helmholtz_op = HelmholtzOperator(**HELMHOLTZ_PARAMS, device=device)

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
        x = torch.stack([template, target], dim=1)  # Stack along the channel dimension
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



def main():
    # Generate the target and template images
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sz = 128
    shape = (sz, sz)
    #template, target = get_target_and_template(device=device, shape=shape)

    template, target = get_target_and_template_from_dataset(idx=None, device=device)


    # Perform diffeomorphic registration.
    # RegistrationCNN is a small, fixed-receptive-field network that tends to fail as resolution increases.
    # UNet has a much larger receptive field/capacity via its encoder-decoder structure and skip connections,
    # so it copes better at higher resolutions. Swap model_cls/model_kwargs below to try either.
    velocity_field, deformed_template = diffeomorphic_registration(
        template, target,
        num_epochs=1000, learning_rate=1e-2, lambda_reg=1e-8, device=device,
        model_cls=UNet, model_kwargs={"base_channels": 32, "depth": 4},
    )

    # Visualize the results. Plotting code lives in test/test_visualization.py.
    from test.test_visualization import plot_deformation_sequence_gif, plot_registration_summary

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

if __name__ == "__main__":
    main()