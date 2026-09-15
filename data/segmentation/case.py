import os
import SimpleITK as sitk
from image_reader import ImageReader, FILENAME_EXTENSIONS, NIFTI
from tqdm import tqdm
from pathlib import Path
from totalsegmentator.map_to_binary import class_map
import numpy as np
from nibabel import Nifti1Image
from utils import to_mu
from tissue import Tissue
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from data.parameters import ALL_SEGMENTATION_TASKS, SEGMENTATION_TASKS, BODY_SEGMENTATION_TASKS, SEGMENTATION_CLASS_MAP, BODY_LABEL, EFFECTIVE_KEV


class Case:
    
    def __init__(self, dir, ct_filename, CT_in_HU=True,
                 read_slices=None, read_midslice_only=False, crop_to_organ=[], 
                 resample=False, resample_args={}, 
                 ):

        self.effective_kev = EFFECTIVE_KEV # keV
        self.dir = dir
        self.ct_filename = ct_filename

        self.read_slices = read_slices
        self.read_midslice_only = read_midslice_only
        self.continuous_slices = not (isinstance(read_slices, list) and len(read_slices)>0)
        self.CT_in_HU = CT_in_HU
        self.crop_to_organ = crop_to_organ
        
        self.segmentation_tasks = SEGMENTATION_TASKS
        self.segmentation_class_map = SEGMENTATION_CLASS_MAP
        self.body_segmentation_task = BODY_SEGMENTATION_TASKS
        self.body_label = BODY_LABEL

        if crop_to_organ:
            self.organ_fov = self.get_organ_axial_field_of_view(organ=crop_to_organ) # [start_idx, end_idx]
        else:
            self.organ_fov = None

        # Initialize the image reader
        self.reader = ImageReader(self, read_slices=self.read_slices, 
                                  read_midslice_only=self.read_midslice_only, 
                                  crop_fov=self.organ_fov,
                                  resample=resample,
                                  resample_args=resample_args)

        # Read CT
        self.read_CT(Path(self.dir, self.ct_filename))

        # Convert to linear attenuation coefficient
        if CT_in_HU:
            self.CT = sitk.Clamp(self.CT, lowerBound=-1000, upperBound=3000)
            self.CT = to_mu(self.CT, self.effective_kev)
            print('converted to mu')
            self.CT_in_HU = False

        # Read segmentations
        self.segmentation, self.segmentation_names = self.read_segmentations(self.segmentation_tasks)
        self.body, self.body_names = self.read_segmentations(self.body_segmentation_task)

        # Mask to body
        self.mask_to_body()

        # Handle the leftovers and overlaps
        self.handle_leftovers()
        #self.handle_overlaps()
        

    def get_organ_axial_field_of_view(self, organ, task='total'):
        segmentation_nifti_img, label_map_dict = load_multilabel_nifti(Path(self.dir, f'{task}.nii.gz'))
        segmentation = segmentation_nifti_img.get_fdata() # numpy
        label_map_inv = {v: k for k, v in label_map_dict.items()}
        if isinstance(organ, list):
            organ_seg = np.zeros_like(segmentation)
            for o in organ:
                try:
                    organ_seg += segmentation == label_map_inv[o]
                except:
                    print(f'Organ {o} not found in the segmentation')
        else:
            organ_seg = segmentation == label_map_dict[organ]
        mask_projection = np.any(organ_seg, axis=(1, 2)) # project to 1D
        # Find indices where the mask is present
        mask_indices = np.where(mask_projection)[0]
        # First and last slice where mask appears
        return [mask_indices[0], mask_indices[-1]]
    
    def mask_to_body(self):
        """ Mask the CT to the body segmentation"""
        #print('mu sum outside body:', sitk.GetArrayFromImage(sitk.Mask(self.CT, self.body == 0, outsideValue=0)).sum())
        body_names_inv = {v: k for k, v in self.body_names.items()}
        self.CT = sitk.Mask(self.CT, self.body == body_names_inv[self.body_label], outsideValue=0)
        #print('mu sum outside body:', sitk.GetArrayFromImage(sitk.Mask(self.CT, self.body == 0, outsideValue=0)).sum())

    def read_CT(self, path):
        """ Read CT nifti in the directory """
        self.reader.set_path(path)
        self.CT = self.reader.read()
        
        
    def read_segmentations(self, tasks, **kwargs):
        """ Read segmentations in the directory and save to dict.
         Note! that the segmentations are added using sitk.Maximum, hence the 
         order of addition matters. For non-overlapping segmentations, the order does not matter."""
        
        if len(tasks) == 0:
            raise ValueError("No tasks given.")
        
        relabeled_segmentation = sitk.Image(self.CT.GetSize(), sitk.sitkUInt8)
        #relabeled_segmentation = Nifti1Image(relabeled_segmentation, self.CT.affine)
        relabeled_segmentation.CopyInformation(self.CT)
        label_names = {}

        for task_idx, task in enumerate(tqdm(tasks, desc=f"Reading segmentations")):
            print(f'Loading segmentations from {task}...')
            #if self._get_next_organ_number()>10: continue
            self.reader.set_path(Path(self.dir, f'{task}.nii.gz'))
            segmentation = self.reader.read()

            classes = self.segmentation_class_map[task]
 
            # Relabel the segmentation image
            unique_labels = sitk.GetArrayViewFromImage(segmentation).astype(int)
            unique_labels = np.unique(unique_labels)
            new_label_map = {label: idx + len(label_names) for idx, label in enumerate(unique_labels) if label != 0}
            
            for current_label, new_label in tqdm(new_label_map.items(), desc='Remapping labels...'):
                relabeled_segmentation = sitk.Maximum(relabeled_segmentation, sitk.Cast(segmentation == current_label, segmentation.GetPixelID()) * new_label)
                label_names[new_label] = classes[current_label]
        
        segmentation = relabeled_segmentation
        segmentation_names = label_names

        return segmentation, segmentation_names

    def _get_next_organ_number(self):
        return len(self.segmentation_names) + 1
    
    def handle_overlaps(self):
        """ Handle overlaps between the multiple segmentations """
        print('Handling overlaps...')
        
        vesselidx = self._get_number_from_name(self.segmentation_names, 'lung_vessels')
        tracheaidx = self._get_number_from_name(self.segmentation_names, 'lung_trachea_bronchia')
        vessel_mask = self.segmentation == vesselidx
        trachea_mask = self.segmentation == tracheaidx
        for lungtissue in ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]:
            lungtissue_label = self._get_number_from_name(self.segmentation_names, lungtissue)
            lungtissue_mask = self.segmentation == lungtissue_label
            vessel_overlap = sitk.And(lungtissue_mask, vessel_mask)
            trachea_overlap = sitk.And(lungtissue_mask, trachea_mask)
            self.segmentation = self.segmentation * sitk.Not(vessel_overlap) * sitk.Not(trachea_overlap)
            self.segmentation = sitk.Add(self.segmentation, vessel_overlap * vesselidx)
            self.segmentation = sitk.Add(self.segmentation, trachea_overlap * tracheaidx)
            

    def handle_leftovers(self):
        """ Assign the parts without segmentation to the residual tissue type """
        print('Handling leftovers...')

        residual = (self.body > 0) - sitk.Cast(self.segmentation > 0, self.body.GetPixelID()) # get the residual

        mu_muscle = Tissue('muscle').linear_att_coeff(self.effective_kev)
        
        residual_volume = sitk.Mask(self.CT, residual, outsideValue=0)
        residual_blood = sitk.Threshold(residual_volume, lower=mu_muscle*1.1, upper=2) > 0
        residual_muscle = sitk.Mask(residual - residual_blood, residual, outsideValue=0)

        muscle_label = self._get_number_from_name(self.segmentation_names, 'skeletal_muscle')
        blood_label = self._get_number_from_name(self.segmentation_names, 'blood')

        if muscle_label is None:
            muscle_label = self._get_next_organ_number()
            self.segmentation_names[muscle_label] = 'skeletal_muscle'
        if blood_label is None:
            blood_label = self._get_next_organ_number()
            self.segmentation_names[blood_label] = 'blood'
        

        self.segmentation = sitk.Cast(self.segmentation, sitk.sitkUInt8)
        self.segmentation = sitk.Add(self.segmentation, sitk.Cast(residual_muscle > 0, sitk.sitkUInt8) * muscle_label)
        self.segmentation = sitk.Add(self.segmentation, sitk.Cast(residual_blood > 0, sitk.sitkUInt8) * blood_label)

      
    @staticmethod
    def _get_number_from_name(d, target_value):
        for key, value in d.items():
            if value == target_value:
                return key
        else:
            return None # Return None if the value is not found




if __name__ == '__main__':
    
    case = Case(dir='../kits23/case_00000', ct_filename='imaging.nii.gz', read_slices=None, read_midslice_only=True)
    print(case.segmentation_names, case.body_names)
   

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(131)
    plt.imshow(sitk.GetArrayFromImage(case.CT)[...,0])
    plt.subplot(132)
    plt.imshow(sitk.GetArrayFromImage(case.segmentation)[...,0])
    plt.subplot(133)
    plt.imshow(sitk.GetArrayFromImage(case.body)[...,0])
    plt.show()
