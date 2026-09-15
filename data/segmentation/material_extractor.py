import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter
from tissue import Tissue, Atom, Bone, Fat, TissueAtomicCompositionTable
from utils import *

from case import Case

from data.parameters import SEGMENTATION_TASKS, BODY_SEGMENTATION_TASKS, CT_FILENAME

class MaterialExtractor:
    def __init__(self, extraction_materials, skip_tissue_type, atomic_composition_table: TissueAtomicCompositionTable):
        self.extraction_materials = extraction_materials
        self.skip_tissue_type = skip_tissue_type
        self.atomic_composition_table = atomic_composition_table

    def set_case(self, case):
        self.case = case
        self.segmentation = case.segmentation

    def apply_noise_to_attenuation(self, attenuation, shape):
        """ Apply noise to the attenuation values """
        noise = np.random.normal(1, 0.05, shape)
        noise = gaussian_filter(noise, 1)
        return attenuation * noise


class SubtractionExtractor(MaterialExtractor):
    """
    Extracts material-specific maps from CT images by subtracting the attenuation contributions of specified materials or tissues.
    This extractor is designed to generate virtual non-contrast (VNC) images or other material-decomposed images by removing the contribution of selected atoms or tissues from the CT data, based on segmentation and atomic composition information.
    Parameters
    ----------
    extraction_materials : list[Atom] | Atom | Tissue
        The materials (atoms or tissues) to extract or subtract from the CT image.
    atomic_composition_table : dict
        Table mapping tissue/material names to their atomic composition.
    material_fractions : dict, optional
        Dictionary mapping organ/tissue names (lowercase) to arrays of fractions for each material. Fractions are normalized to sum to 1.
    skip_atom : list[Atom] | Atom, optional
        Atoms to skip during extraction.
    skip_tissue_type : type | tuple[type], optional
        Tissue types to skip during extraction.
    clip_values : tuple(float | None, float | None), optional
        Minimum and maximum values to clip the resulting material maps.
    filter_result : bool, default False
        Whether to apply post-processing filtering to the extracted material maps.
    dilated_tissue_type_to_remove : type, optional
        Tissue type (e.g., Bone) to dilate and remove from the material maps to avoid leakage.
    Attributes
    ----------
    n_materials : int
        Number of materials to extract.
    material_fractions : dict
        Normalized material fractions for each tissue.
    clip_values : tuple
        Clipping range for output values.
    filter_result : bool
        Whether to filter the resulting material maps.
    dilated_tissue_type_to_remove : type or None
        Tissue type to remove by dilation.
    Methods
    -------
    extract(image=None)
        Extracts the subtraction of the segmentations, returning material maps and the leftover image.
    remove_dilated_tissue_type(material, type_to_remove=Bone)
        Dilates and removes specified tissue type from the material map.
    filter_material_map(material)
        Applies filtering and masking to the material map.
    """


    
    def __init__(self, extraction_materials, atomic_composition_table, material_fractions={}, 
                 skip_atom=None, skip_tissue_type=None, clip_values=(None, None), filter_result=False,
                 dilated_tissue_type_to_remove=None):
        super().__init__(extraction_materials, skip_tissue_type, atomic_composition_table)
        #assert isinstance(self.extraction_material, Atom), "Subtraction atom must be of type Atom"

        if isinstance(self.extraction_materials, Tissue):
            self.skip_atom = [Atom(a) for a in self.extraction_materials.atoms]
        elif isinstance(self.extraction_materials, Atom):
            self.skip_atom = [self.extraction_materials.name]
        elif isinstance(self.extraction_materials, list):
            self.skip_atom = [a.name for a in self.extraction_materials]

        if isinstance(skip_atom, list):
            self.skip_atom.append(*skip_atom)
        elif isinstance(skip_atom, Atom):
            self.skip_atom.append(skip_atom)

        self.n_materials = len(self.extraction_materials)
        self.material_fractions = material_fractions
        self.material_fractions = {k.lower(): v / np.sum(v) for k, v in material_fractions.items()} # normalize to sum to 1 and lower case

        self.clip_values = clip_values

        self.filter_result = filter_result

        self.dilated_tissue_type_to_remove = dilated_tissue_type_to_remove

    def extract(self, image=None):
        """ Extract the subtraction of the segmentations """

        image = self.case.CT if image is None else image

        ct_array = sitk.GetArrayFromImage(image)
        segmentation = sitk.GetArrayFromImage(self.segmentation)
        materials = np.zeros((self.n_materials, *ct_array.shape), dtype=ct_array.dtype)
        
        for organ_number, organ_name in self.case.segmentation_names.items():

            tissue = Tissue(organ_name, self.atomic_composition_table)
            
            if 'lobe' in organ_name.lower():
                print('Found lung lobe, skipping', organ_name, tissue.type)
                continue
             
            if self.skip_tissue_type is not None and isinstance(tissue.type, tuple(self.skip_tissue_type)):
                print('Skipping', organ_name)
                continue

            if organ_name.lower() in self.material_fractions.keys(): # check if current organ name is in given material fractions list
                material_weights = self.material_fractions[organ_name.lower()]
            else:
                continue # if no weights are given, assume no material contribution

            organ_local_density = segmentation == organ_number # could be updated to allow fractions
            attenuation = tissue.linear_att_coeff(self.case.effective_kev, skip_atom=self.skip_atom)

            organ_local_density = segmentation == organ_number # could be updated to allow fractions
            attenuation = tissue.linear_att_coeff(self.case.effective_kev, skip_atom=self.skip_atom)

            # if tissue.is_lesion:
            #     attenuation = self.apply_noise_to_attenuation(attenuation, organ_local_density.shape)

            attenuation = attenuation * organ_local_density # construct attenuation map for the organ
            organ_material = ct_array[organ_local_density] - attenuation[organ_local_density] # subtract to get material contribution within the organ

            material_weights = np.array(material_weights)
            material_weights = material_weights / material_weights.sum()
            material_weights = material_weights.reshape(self.n_materials, *(1,)*len(organ_material.shape))    
            organ_material = np.stack([organ_material] * self.n_materials, axis=0)
            organ_material = material_weights * organ_material
            attenuation = attenuation * organ_local_density
            
            materials[:, organ_local_density] = organ_material.clip(*self.clip_values)

        materials_ = []
        leftovers = self.case.CT

        for m in range(materials.shape[0]):

            material = materials[m]
            material = sitk.GetImageFromArray(material)
            material.CopyInformation(self.case.CT)

            print(material.GetSize(), material.GetSpacing(), material.GetOrigin(), material.GetDirection())
            if self.filter_result:
                material = self.filter_material_map(material)

            if self.dilated_tissue_type_to_remove is not None:
                print('Removing dilated tissue type', self.dilated_tissue_type_to_remove)
                material = self.remove_dilated_tissue_type(material, self.dilated_tissue_type_to_remove)

            leftovers -= material # calculate what is left: i.e. the VNC when subtracting iodine

            material /= self.extraction_materials[m].linear_att_coeff(self.case.effective_kev) # divide by mu to get 'a' in unitless concentration

            materials_.append(material)
        
        return materials_, leftovers
    
    def remove_dilated_tissue_type(self, material, type_to_remove=Bone):
        """ Dilate bone masks and remove them from the material map to avoid bone leakage """
        for organ_number, organ_name in self.case.segmentation_names.items():
            tissue = Tissue(organ_name, self.atomic_composition_table)
            print(tissue.type)
            if isinstance(tissue.type, type_to_remove):
                bone_map = self.segmentation == organ_number
                kernel = [1] * 3
                if not self.case.continuous_slices:
                    kernel[ImageReader.z_axis_index(self.case.CT)] = 0 # set z dilate to 0
                bone_map = sitk.BinaryDilate(bone_map, kernel)
                material = sitk.Mask(material, sitk.Not(bone_map), outsideValue=0)
        return material

    
    def filter_material_map(self, material):
         #material = sitk.Bilateral(material, .15, 10, 10)
        material_mask = material>0
        material = sitk.Mask(material, material_mask, outsideValue=0)
        
        kernel = [0.5] * 3
        if not self.case.continuous_slices:
            kernel[self.case.z_axis_index] = 0 # set z blurring to 0
        material = sitk.DiscreteGaussian(material, variance=kernel)
            
        material = sitk.Mask(material, material_mask, outsideValue=0) # mask the blurred materials to the right part
        return material

    
class ChangeOfBasisExtractor(MaterialExtractor):
    """
    Extracts material basis images from CT data using a change-of-basis approach.
    This extractor computes the local densities of a set of basis materials for each voxel in a CT image,
    based on their linear attenuation coefficients at multiple energies. It uses a pseudo-inverse of the
    attenuation matrix to solve for the material weights, optionally skipping specified atoms or tissue types.

    Parameters
    ----------
    extraction_materials : list[Atom] | list[Tissue]
        The materials (atoms or tissues) to use as the basis for decomposition. Must contain more than one material.
    atomic_composition_table : dict
        Table mapping material names to their atomic compositions.
    skip_atom : str | Atom, optional
        Atom type to skip when computing attenuation coefficients.
    skip_tissue_type : type | tuple[type], optional
        Tissue types to skip during extraction. Voxels of this type are assigned to the first material.

    Attributes
    ----------
    n_materials : int
        Number of basis materials.
    energies : np.ndarray
        Array of energies (in keV) at which attenuation coefficients are evaluated.
    skip_atom : str | Atom
        Atom type to skip.
    extraction_materials : list
        List of basis material objects.
    atomic_composition_table : dict
        Atomic composition table.
    skip_tissue_type : type | tuple[type]
        Tissue type(s) to skip.

    Methods
    -------
    extract(image=None)
        Performs the material decomposition on the provided CT image (or the default case image if not provided).
        Returns a list of SimpleITK images, each representing the local density of a basis material.
    """
        
    def __init__(self, extraction_materials, atomic_composition_table, skip_atom=None, skip_tissue_type=None):
        super().__init__(extraction_materials, skip_tissue_type, atomic_composition_table)
        assert isinstance(self.extraction_materials, list), "ChangeOfBasisExtractor.extraction_material must be list of len > 1"
        # assert len(self.extraction_materials) > 1, "ChangeOfBasisExtractor.extraction_material must be list of len > 1"
        self.n_materials = len(self.extraction_materials)
        self.energies = np.linspace(20, 140, self.n_materials)
        self.skip_atom = skip_atom
        print(self.skip_tissue_type)

    def extract(self, image=None):
        
        image = self.case.CT if image is None else image

        M = np.zeros((len(self.energies), self.n_materials))
        for m, material in enumerate(self.extraction_materials):
            M[:, m] = material.linear_att_coeff(self.energies)

        effective_mu = [material.linear_att_coeff(self.case.effective_kev) for material in self.extraction_materials]

        ct = sitk.GetArrayFromImage(image)
        segmentation = sitk.GetArrayFromImage(self.segmentation)
                
        M_inv = np.linalg.pinv(M)
        local_densities = np.zeros((self.n_materials, *ct.shape))

        for organ_number, organ_name in self.case.segmentation_names.items():
            tissue = Tissue(organ_name, self.atomic_composition_table)

            if self.skip_tissue_type is not None and isinstance(tissue.type, tuple([self.skip_tissue_type])):
                print('Skipping', organ_name, tissue.type)
                w = np.array([1, 0]) # set all weight to first material (water in most cases). Should be updated with a separate kwarg
            else:
                mu = tissue.linear_att_coeff(self.energies, skip_atom=self.skip_atom)
                w = M_inv @ mu
                print(organ_name, tissue.type)

            organ_local_density = segmentation == organ_number # should be updated to allow fractions

            # correction factor to material weights to align with input image
            w_corr = ct[organ_local_density] / np.dot(w, effective_mu)

            w = w[:, np.newaxis] * w_corr[np.newaxis, :]

            local_densities[:, organ_local_density==1] = w

        materials = []
        for i in range(self.n_materials):
            mat = local_densities[i,...]
            mat = sitk.GetImageFromArray(mat)
            mat.CopyInformation(image)
            materials.append(mat)

        return materials





if __name__ == '__main__':
    from tissue import Bone, Fat
    # from case import Case
    import argparse, os
    from utils import to_HU
    from pathlib import Path
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import torch
    from image_reader import ImageReader
   
    import os

    parser = argparse.ArgumentParser(description='Material extraction from CT images.')
    parser.add_argument('--datadir', type=str, default='/Users/joelva/Documents/datasets/kits23/dataset', help='Directory containing dataset')
    parser.add_argument('--ct_filename', type=str, help='The filename of the input ct file', default=CT_FILENAME)    
    parser.add_argument('--savedir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--cases', type=str, default='0', help='Slice or list of cases to process, e.g. "0:10" or "1,2,3"')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    args = parser.parse_args()

    datadir = args.datadir
    savepath = args.savedir if args.savedir is not None else datadir

    subdirs = sorted([d for d in os.listdir(datadir) if Path(datadir, d).is_dir()]) # sort dirs
    
    if ':' in args.cases:
        start, end = args.cases.split(':')
        cases = slice(int(start) if start else None, int(end) if end else None)
    elif ',' in args.cases:
        cases = [int(x) for x in args.cases.split(',')]
    else:
        cases = slice(int(args.cases), int(args.cases)+1)
    subdirs = subdirs[cases] # extract cases of interest
    plot = args.plot
    print(subdirs)
    for case_num in tqdm(subdirs, desc='Extracting materials from cases...'):
        print('Processing', case_num)

        subdir = Path(datadir, case_num)
        savedir = Path(savepath, case_num)

        os.makedirs(savedir, exist_ok=True)
        print(subdir, savedir)
        case = Case(subdir, ct_filename=args.ct_filename, read_midslice_only=False, read_slices=None, crop_to_organ=['kidney_left', 'kidney_right'])
        print(case.CT.GetSize(), case.CT.GetSpacing(), case.CT.GetOrigin(), case.CT.GetDirection())
        atomicCompTable = TissueAtomicCompositionTable()
        subtractionExtractor = SubtractionExtractor([Atom('I', density=0.02), Atom('Gd', density=0.004)], 
                                                    atomicCompTable, 
                                                    material_fractions={ 'kidney_left':[.9,.1], 
                                                                    'kidney_right':[.9,.1],
                                                                    'aorta':[.8,.2],
                                                                    'liver':[.85,.15],
                                                                    'blood':[.8,.2],
                                                                    'cartilage':[.2,.8],
                                                                   }, 
                                                    clip_values=(0, None), 
                                                    skip_tissue_type=[Bone, Fat], 
                                                    filter_result=True, 
                                                    dilated_tissue_type_to_remove=Bone)
        subtractionExtractor.set_case(case)
        subtracted_materials, vnc = subtractionExtractor.extract()
        
        # twoBasisExtractor = ChangeOfBasisExtractor([Tissue('water'), Atom('Ca')], atomicCompTable)
        
        twoBasisExtractor = ChangeOfBasisExtractor([Tissue('water')], atomicCompTable)
        twoBasisExtractor.set_case(case)
        twoBasis_materials = twoBasisExtractor.extract(vnc)

        for i, mat in enumerate(subtracted_materials):
            sitk.WriteImage(mat, Path(savedir, f'{subtractionExtractor.extraction_materials[i].name.lower()}.nii.gz'))
        for i, mat in enumerate(twoBasis_materials):
            sitk.WriteImage(mat, Path(savedir, f'{twoBasisExtractor.extraction_materials[i].name.lower()}.nii.gz'))

        sitk.WriteImage(case.segmentation, Path(savedir, 'segmentation.nii.gz'))
        torch.save({'segmentation': case.segmentation_names, 'body': case.body_names, 'indices': case.organ_fov}, Path(savedir, 'labels.pt'))


        if plot:
            # Plot the result to the subdir
            print('Plotting results...')

            z_index = ImageReader.get_z_index(case.CT)
            slice_number = case.CT.GetSize()[z_index] // 2
            slice_obj = [slice(None)] * 3
            slice_obj[z_index] = slice_number
        
            images = [sitk.GetArrayFromImage(case.CT[slice_obj]), sitk.GetArrayFromImage(case.segmentation[slice_obj])]
            titles = ['Input CT', 'Segmentation']

            for i, mat in enumerate(subtracted_materials):
                im = sitk.GetArrayFromImage(mat[slice_obj])
                print(im.shape)
                images.append(im)
                titles.append(f'{subtractionExtractor.extraction_materials[i].name} {subtractionExtractor.extraction_materials[i].density:.2g} g/cm³')
            
            for i, mat in enumerate(twoBasis_materials):
                im = sitk.GetArrayFromImage(mat[slice_obj])
                images.append(im)
                titles.append(f'{twoBasisExtractor.extraction_materials[i].name} {twoBasisExtractor.extraction_materials[i].density:.2g} g/cm³')
    

            n_images = len(images)
            n_cols = int(np.ceil(np.sqrt(n_images)))
            n_rows = int(np.ceil(n_images / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
            axs=axs.flatten()
            print(axs.shape)
            for i, im in enumerate(images):
                im=axs[i].imshow(im.squeeze().T, cmap='gray')
                axs[i].set_title(titles[i])
                plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
                
            for ax in axs: ax.axis('off')
            #plt.tight_layout()
            plt.savefig(Path(savedir, f'spectral_phantom_result.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
