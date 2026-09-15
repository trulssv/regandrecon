import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib
from utils import resample

NIFTI = 'nifti'
DICOM = 'dicom'

FILENAME_EXTENSIONS = {NIFTI: ('.nii', '.nii.gz'), 
                       DICOM: ('.dcm')}

class ImageReader:
    def __init__(self, path=None, read_slices=None, read_midslice_only=False, crop_fov=[], resample=False, resample_args=None):
        self.path = path
        self.im = None
        self.read_slices = read_slices
        self.read_midslice_only = read_midslice_only
        self.crop_fov = crop_fov
        if read_slices is not None:
            assert read_midslice_only is False, "Cannot read given slices and midslice only at the same time."

        self.resample = resample
        if resample:
            self.resample_args = resample_args
            assert self.resample_args['spacing'] is not None, 'Spacing must be set for resampling'
            assert self.resample_args['size'] is not None, 'Size must be set for resampling'
            assert self.resample_args['save'] is not None, 'Save path must be set for resampling'

    def set_path(self, path):
        self.path = path
        self.image_type = self._get_image_type()
        self.im = None # reset the image

    @staticmethod
    def get_z_index(image):
        """ Get the z axis index of the CT image """
        try:
            size = image.GetSize()
        except:
            size = image.shape

        if size[0] == size[1]:
            z_index = 2
        elif size[0] == size[2]:
            z_index = 1
        elif size[1] == size[2]:
            z_index = 0
        else:
            UserWarning('Cannot determine z axis index due to different pixels in axial plane, defaulting to 0')
            z_index = 0
        return z_index
    
    def _resample(self, image):
        """ Resample the image to the given spacing and size """
        new_spacing = [self.resample_args['spacing']]*3
        new_size = [self.resample_args['size']]*3
        new_image = resample(image, new_spacing, new_size, **self.resample_args.get('kwargs'))
        return new_image
        
    def read(self):
        """Read the image from the path and return as a SimpleITK image."""
        # Load NIfTI image using SimpleITK
        sitk_im = sitk.ReadImage(self.path)

        if self.resample:
            # Resample the image
            sitk_im = self._resample(sitk_im)
  
        # Get original image spacing, origin, and direction
        spacing = sitk_im.GetSpacing()  # (x, y, z) spacing
        origin = sitk_im.GetOrigin()
        direction = sitk_im.GetDirection()
        #print(spacing, sitk_im.GetSize())

        # Convert to NumPy array
        img_array = sitk.GetArrayFromImage(sitk_im)  # Shape: (slices, height, width)

        # Find axial index based on numpy array
        z_index = ImageReader.get_z_index(img_array) 

        # Crop Field of View (FOV) if specified
        if self.crop_fov:
            if z_index == 0:
                img_array = img_array[self.crop_fov[0]:self.crop_fov[1], :, :]
            elif z_index == 1:
                img_array = img_array[:, self.crop_fov[0]:self.crop_fov[1], :]
            elif z_index == 2:
                img_array = img_array[:, :, self.crop_fov[0]:self.crop_fov[1]]

        # Read only the middle slice
        n_slices = img_array.shape[z_index]
        if self.read_midslice_only:
            if z_index == 0:
                img_array = img_array[n_slices//2:n_slices//2 + 1, :, :]
            elif z_index == 1:
                img_array = img_array[:, n_slices//2:n_slices//2 + 1, :]
            elif z_index == 2:
                img_array = img_array[:, :, n_slices//2:n_slices//2 + 1]

        # Read specific slices based on percentages
        elif isinstance(self.read_slices, list):
            indices = []
            for s in self.read_slices:  # Convert percentage to slice index
                assert 0 <= s <= 1, "Slice percentage should be between 0 and 1."
                indices.append(int((n_slices - 1) * s))
            if z_index == 0:    
                img_array = img_array[indices, :, :]
            elif z_index == 1:
                img_array = img_array[:, indices, :]
            elif z_index == 2:
                img_array = img_array[:, :, indices]
                
        # Convert back to SimpleITK image
        self.im = sitk.GetImageFromArray(img_array)

        # Restore original spacing, origin, and direction
        self.im.SetSpacing(spacing)
        self.im.SetOrigin(origin)
        self.im.SetDirection(direction)

        return self.im


    def nib_read(self):
        """ Read the image from the path and return as sitk image """
        nib_im = nib.load(self.path)
        header = nib_im.header
            
        # find axial index
        indices = nib.aff2axcodes(nib_im.affine)
        z_index = indices.index('I') if 'I' in indices else indices.index('S')
        slicer = [slice(None)] * len(nib_im.shape)
        
        if self.crop_fov:
            if z_index   ==0: nib_im = nib_im.slicer[self.crop_fov[0]:self.crop_fov[1],:,:]
            elif z_index ==1: nib_im = nib_im.slicer[:,self.crop_fov[0]:self.crop_fov[1],:]
            elif z_index ==2: nib_im = nib_im.slicer[:,:,self.crop_fov[0]:self.crop_fov[1]]
        
        n_slices = nib_im.shape[z_index]

        if self.read_midslice_only:
            slicer[z_index] = slice(n_slices//2, n_slices//2 + 1)
            nib_im = nib_im.slicer[tuple(slicer)]
            nib_im = np.asanyarray(nib_im.dataobj)
        elif isinstance(self.read_slices, list):
            nib_im = np.asanyarray(nib_im.dataobj)
            indices=[]
            for s in self.read_slices: # percentages of n_slices
                assert 0 <= s <= 1, "Slice percentage should be between 0 and 1."
                indices.append(int((n_slices-1) * s))
            if z_index   ==0: nib_im = nib_im[indices,:,:]
            elif z_index ==1: nib_im = nib_im[:,indices,:]
            elif z_index ==2: nib_im = nib_im[:,:,indices]

        # Convert to SimpleITK image
        self.im = sitk.GetImageFromArray(nib_im)

        # Copy the metadata from the nibabel image to the SimpleITK image
        for key in header.keys():
            self.im.SetMetaData(key, str(header[key]))
        
        return self.im
    
    def get_metadata(self):
        """ Read the metadata of the image """
        return {k: self.im.GetMetaData(k) for k in self.im.GetMetaDataKeys()}
    
    def _get_image_type(self):
        """ Return the name of the image type (nifti, dicom) """
        if os.path.isfile(self.path):
            for key, values in FILENAME_EXTENSIONS.items():
                if any(ext in self.path.suffixes for ext in values):
                    return key
        elif os.path.isdir(self.path):
            return DICOM


if __name__ == '__main__':
    from pathlib import Path
    path = Path('/Users/joelva/Documents/datasets/kits23/dataset/case_00000/imaging.nii.gz')
    print(path.suffixes)
    reader = ImageReader()
    reader.set_path(path)
    image = reader.read()
    print(image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())
    meta = reader.get_metadata()
    print(image, meta)
