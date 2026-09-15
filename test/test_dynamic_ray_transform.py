import odl
import torch
import numpy as np
from matplotlib import pyplot as plt
from sigpy import shepp_logan

def generate_circle2D():

    """This function generates a simple 2D circle phantom for testing the ray transform."""
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    radius = 0.5
    Z[X**2 + Y**2 <= radius**2] = 1.0
    return Z.astype(np.float32)

def generate_rectangle2D():

    """This function generates a simple 2D rectangle phantom for testing the ray transform."""
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    Z[np.logical_and(np.abs(X) <= 0.5, np.abs(Y) <= 0.2)] = 1.0
    return Z.astype(np.float32)



# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 128 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[128, 128], dtype='float32')

# Angles: uniformly spaced, n = 200, min = pi/6, max = pi/4
angle_partition = odl.uniform_partition(np.pi/2, np.pi, 200)

# angle_partition = odl.uniform_partition(0, np.pi / 6, 200)


# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 500)

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)


# --- Create Filtered Back-projection (FBP) operator --- #


# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Adjoint of the ray transform (back-projection).
adjoint_trafo = ray_trafo.adjoint

# FBP operator with a Ram-Lak filter.
fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Ram-Lak')


# --- Test the operators --- #

# Create a simple 2D rectangle phantom
phantom = generate_rectangle2D()

# Shepp-Logan phantom for testing
phantom = shepp_logan((128, 128)).astype(np.float32)

# Compute the sinogram (forward projection)
sinogram = ray_trafo(phantom)

print("Phantom shape:", phantom.shape)
print("Sinogram shape:", sinogram.shape)

# Compute the adjoint (back-projection)
adjoint = adjoint_trafo(sinogram)

# Compute the FBP reconstruction
reconstruction = fbp_op(sinogram)

# Visualize the results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('Phantom')
axes[0].axis('off') 
axes[1].imshow(sinogram, cmap='gray')
axes[1].set_title('Sinogram')
axes[1].axis('off')
axes[2].imshow(adjoint, cmap='gray')
axes[2].set_title('Adjoint (Back-projection)')
axes[2].axis('off')
axes[3].imshow(reconstruction, cmap='gray')
axes[3].set_title('FBP Reconstruction')
axes[3].axis('off')
plt.show()

