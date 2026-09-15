import torch
import numpy as np
from matplotlib import pyplot as plt

# PATHS

import os
from pathlib import Path

# DEFORMATION OPERATORS

from operators.lddmm.deform import GroupAction, VelocityIntegrator, FlowDeformationOperator
from sigpy import shepp_logan

PLOT_DIR = Path("test/plots")

### LOAD Shepp-Logan phantom image

phantom = torch.from_numpy(shepp_logan((128, 128), dtype=np.float32))

def _get_velocity_field():
    """This function generates a simple velocity field for testing purposes. The velocity field is a 2D vector field that defines the deformation of the image. 
    The deformation_magnitude parameter controls the strength of the deformation."""

    v = torch.zeros((1, 128, 128, 2), dtype=torch.float32)

    deformation_magnitude = 50.0  # Adjust this value to increase or decrease the strength of the deformation

    # Create a simple velocity field that deforms the image in a circular pattern
    for i in range(128):
        for j in range(128):
            # Compute the distance from the center of the image
            x = i - 64
            y = j - 64
            distance = torch.sqrt(torch.tensor(x**2 + y**2, dtype=torch.float32)).clamp(min=1)

            # Define the velocity field based on the distance from the center
            if distance > 0:
                v[0, i, j, 0] = -deformation_magnitude * y / distance  # x component of velocity
                v[0, i, j, 1] = deformation_magnitude * x / distance   # y component of velocity
    print("Generated velocity field with shape:", v.shape)

    return v

v = _get_velocity_field()

print("Velocity Field:", v.shape)

group_action = GroupAction(phantom.shape)
velocity_integrator = VelocityIntegrator(10, phantom.shape)
deform_op = FlowDeformationOperator(phantom.shape, 7)

deformed_phantom = deform_op(phantom.unsqueeze(0).unsqueeze(0), v)

print("Original Phantom Shape:", phantom.shape)
print("Deformed Phantom Shape:", deformed_phantom.shape)
print("Velocity Field Shape:", v.shape)

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].imshow(phantom.squeeze(), cmap="gray")
ax[0].set_title("Original Phantom")
ax[1].imshow(deformed_phantom[0, -1].squeeze().detach().numpy(), cmap="gray")
ax[1].set_title("Deformed Phantom")
ax[2].imshow(torch.sqrt(v[0, :, :, 1] ** 2), cmap="hot")
ax[2].set_title("Velocity Field Magnitude") 


os.makedirs(PLOT_DIR, exist_ok=True)
print("Saving deformed phantom image to:", PLOT_DIR / "deformed_phantom.png")
plt.savefig(PLOT_DIR / "deformed_phantom.png")














