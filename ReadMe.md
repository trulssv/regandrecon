# Simultaneous Registration and Reconstruction with deep learning in CT

This repo contains a pipeline for training deep learning models for performing simultaneous registration and reconstruction. The repo consists of the following parts:


1. data
2. operators
3. model
4. training
5. visualization

## data

The folder contains all functionality for loading and preprocessing data for training. This consists of the following steps:

#### Preprocessing pipline

###### Create initial data loader

This is the first step of preprocessing. A data loader is constructed that loads the raw data. The purpose of this is to perform a robust train/test/val split and possibly remove corrupt samples.

###### Preprocess volumes

In this step, the ground truth volumes are preprocessed. First, spatio-temporal 4D volumes will be formed from individual 3D volumes. Then, the data will be normalized from HU to [0, 1], and reshaped to N x T x D x H x W i.e. (b, t, z, y, x). 

###### Denoising / motion artifact correction

In this step we perform denoising using the open source DRUNet. This is not ideal, since it is trained for Gaussian Noise on 2D slices and not 3D volumetric CT data with Poisson noise, but provides a simple plug-and-play framework.

###### CT data simulation

In this step

1. A ray transform instance with provuded geometry in ODL is created
2. Data is forward projected to normalized pathlenghts
3. Noise is added by combining the initial flux with normalized pathlenghts to obtain the Poisson parameters


## Operators

The following operators need to be implemented:

1. Ray Transform $\mathcal{T}$
2. Diffeomorphic Group Action:  $\mathcal{V}_{\phi}$
3. Velocity Integration Operator: $\Psi$
4. LDDMM Operator: $L = (\operatorname{id}\cdot\alpha -\gamma\nabla^2)^\beta$        


### Ray Transform

The Ray Transform is implemented with ODL-astra cuda backend. Two variants are considered:

1. Simple Ray Transform
2. Dynamic Ray Transform


The Simple Ray Transform maps a time sequences of image to a time sequence of sinograms with full angular range. The dynamic ray transform specifies an angular range for each time bin. Hence, each time bin is only partially sampled in the projection domain; giving us a limited angle tomography problem which is more realistic. Both transform supports the following geometries:

1. Parallel3D
2. Cone3D
3. Hellical

The parameters (and geometry) for the Ray Transform are specified in operats/ray_trafo.json

## Previous TODOS

0. Run the full data preprocessing pipline   (Tomorrow) ✓
1. Implement deformation and LDDMM Operators (Tomorrow) ✓
2. Implement HLPD model and CNN Modules      (Tomorrow) ✓
3. Implement training Pipeline                          ✓   

4. Implement quality measures in loss/quality_measures.py (MSE, SSIM, NGF, HelmholtzNorm, VGG16) (Tomorrow)     ✓
5. Implement time conscistency loss                                                              (Tommorow)     ✓
6. Add further regularization terms to the loss if needed                                        (Tommorow)     ✓      
7. Train and get a first model for benchmarking                                                  (Tommorow)     ✓
8. Create an eval script                                                                         (Tommorow)     ✓

## TODOS

0. Write a summary script (QM / epochs)                  (Tomorrow)             ✓
1. Benchmark algorithm (what takes time and how much)    (Tomorrow)
2. Optimize the algorithm                                (Tomorrow)
3. Improve on previous run                               (Tomorrow)


6. Train with temporal superresolution
7. Hyperparameter grid search
8. Implement conventional motion compensation schemes for comparrison







