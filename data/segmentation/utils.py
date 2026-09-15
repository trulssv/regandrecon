import SimpleITK as sitk
import numpy as np
import nibabel as nib
import torch
import os
from scipy.ndimage import gaussian_filter

def pre_process_volume(vol, rescale="0-1", to_numpy=False):
    """" Preprocesses a 3D CT volume.  """""

    if to_numpy:
        vol = vol.numpy(force=True) # Convert from torch tensor to numpy array
        vol = vol.astype(np.float32)

    # TODO apply denoising and artifact correction steps in preprocessing.

    if rescale == "hu":
        vol = rescale_hu(vol) # Rescale to HU
    elif rescale == "0-1":
        vol = rescale_0_1(vol)

    return vol

def rescale_hu(vol, min_hu=-1024, max_hu=3072):
    """"Rescales data from [-1, 1] to [min_hu, max_hu]"""""
    vol = 0.5 * (vol + 1) * (max_hu - min_hu) + min_hu
    return vol

def rescale_hu_inv(vol, min_hu=-1024, max_hu=3072):
    """"Normalizes data from [min_hu, max_hu] to [-1, 1]"""""
    vol =  2 * (vol -min_hu) / (max_hu - min_hu) -1
    return vol

def rescale_0_1(vol):
    return 0.5 * (vol +1)

def rescale_0_1_inv(vol):
    return 2 * vol + 1

def ConvertToNifti(path_to_data) -> None:
    """"Loads a pytorch-volume and saves it as a nifti file and rescales data from [-1, 1] to HU"""""

    file = os.path.join(path_to_data, 'volume.pt')
    savefile = os.path.join(path_to_data, 'volume.nii.gz')

  #  if os.path.exists(savefile):
#     return

    data = torch.load(file, weights_only=False).numpy()

    # rescale to HU

    data = rescale_hu(data)

    # convert to nifti

    nifti_data = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti_data, savefile)

def NumpyFromNiftiPath(nifty_path) -> np.ndarray:
    nifti = nib.load(nifty_path)
    numpy = np.array(nifti.dataobj)
    return numpy

def NiftyfromNumpyarray(in_data):
    nifti_data = nib.Nifti1Image(in_data, np.eye(4))
    return nifti_data

def BoneSegmenter(in_data: np.array, sigma:float = 2, bone_hu_threshold:float = 150.0) -> np.array:

    in_data = NumpyFromNiftiPath(in_data)

    # Low pass-filter image

    lowpass = gaussian_filter(in_data, sigma=sigma)

    # use HU threshold ro mask

    bone_seg = (lowpass > bone_hu_threshold).astype(np.float32)

    bone_seg = NiftyfromNumpyarray(bone_seg)

    return bone_seg

def AirSegmenter(in_data: np.array, sigma:float = 20.0, air_hu_threshold:float = -800) -> np.array:
    # Low pass-filter image

    in_data = NumpyFromNiftiPath(in_data)

    lowpass = gaussian_filter(in_data, sigma=sigma)

    # use HU threshold ro mask

    air_seg = (lowpass > air_hu_threshold).astype(np.float32)

    air_seg = NiftyfromNumpyarray(air_seg)

    return air_seg

def bilateral_2d_slices(image, domainSigma=4.0, rangeSigma=50.0, numberOfRangeGaussianSamples=100, axis=0):
    """
    Applies bilateral filtering only in 2D slices along a specified axis.
    """
    # Convert SimpleITK Image to NumPy array
    img_array = sitk.GetArrayFromImage(image)  # numpy shape: (Z, Y, X)
    
    # Apply bilateral filtering to each 2D slice in the selected axis
    for i in range(img_array.shape[axis]):
        if axis == 0:
            img_array[i, :, :] = sitk.GetArrayFromImage(
                sitk.Bilateral(sitk.GetImageFromArray(img_array[i, :, :]), domainSigma, rangeSigma, numberOfRangeGaussianSamples)
            )
        elif axis == 1:
            img_array[:, i, :] = sitk.GetArrayFromImage(
                sitk.Bilateral(sitk.GetImageFromArray(img_array[:, i, :]), domainSigma, rangeSigma, numberOfRangeGaussianSamples)
            )
        elif axis == 2:
            img_array[:, :, i] = sitk.GetArrayFromImage(
                sitk.Bilateral(sitk.GetImageFromArray(img_array[:, :, i]), domainSigma, rangeSigma, numberOfRangeGaussianSamples)
            )

    # Convert back to SimpleITK Image
    filtered_image = sitk.GetImageFromArray(img_array)
    filtered_image.CopyInformation(image)
    return filtered_image


def resample(image, target_spacing, target_size, **kwargs):
    """
    Resamples a CT image to a fixed axial resolution while ensuring a 512x512 in-plane size.
    Also centers the volume in the final FOV.

    Args:
        image (sitk.Image): Input SimpleITK image (assumed shape ZxXxY).
        target_spacing (tuple): Desired spacing (z, x, y). If z is None, it remains unchanged.
        target_size (tuple): Desired output size (depth, width, height). If depth is None, it is computed automatically.

    Returns:
        sitk.Image: Resampled and centered image.
    """
    # Get original properties
    print(image.GetSize(), image.GetSpacing())
    orig_size = np.array(image.GetSize())  # (Z, X, Y)
    orig_spacing = np.array(image.GetSpacing())  # (Z, X, Y) spacing
    orig_origin = np.array(image.GetOrigin())
    orig_direction = image.GetDirection()

    # Set target spacing (keeping original Z spacing if None)
    if target_spacing[0] is None:
        target_spacing = (orig_spacing[0], target_spacing[1], target_spacing[2])

    # Compute new Z size to preserve FOV
    if target_size[0] is None:
        target_size = (
            int(round(orig_size[0] * (orig_spacing[0] / target_spacing[0]))),  # Compute new depth
            target_size[1],  # Keep X as fixed 512
            target_size[2]   # Keep Y as fixed 512
        )

    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputDirection(orig_direction)
    resampler.SetInterpolator(sitk.sitkLinear)  # Use nearest neighbor for segmentation masks
    resampler.SetDefaultPixelValue(0)  # Background value
    resampler.SetTransform(sitk.Transform())

    # Compute new origin to center the volume
    new_origin = orig_origin + (orig_size * orig_spacing - np.array(target_size) * np.array(target_spacing)) / 2.0
    resampler.SetOutputOrigin(tuple(new_origin))

    # Perform resampling
    resampled_image = resampler.Execute(image)
    print(resampled_image.GetSize(), resampled_image.GetSpacing())

    return resampled_image
