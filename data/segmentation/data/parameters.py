import json

from .TS_tissue_types import tissue_types
from totalsegmentator.map_to_binary import class_map

# CHANGE THESE TWO LINES TO USE YOUR OWN DATA
# tissue type grouping, fat, bone etc.
TISSUETYPES = tissue_types
# convert between segmentation name and atomic composition data name
CONVERTDICT = 'TS_to_WW_convert_dict.json'
# table of atomic composition data, has to be in the same format as the Woodard & White table
ATOMIC_COMPOSITION_TABLE = 'woodardwhite.csv'
# the name of the CT file in the case directory
CT_FILENAME = 'imaging.nii.gz'  
# segmentation class map, has to be in the same format as the TotalSegmentator class map
SEGMENTATION_CLASS_MAP = class_map # must contain body class map as well
SEGMENTATION_TASKS = ['total', 'tissue_4_types']
BODY_SEGMENTATION_TASKS = ['body']
ALL_SEGMENTATION_TASKS = SEGMENTATION_TASKS + BODY_SEGMENTATION_TASKS
BODY_LABEL = 'body_trunc'
# effective keV for the CT images, used for attenuation correction
EFFECTIVE_KEV = 70  # should be read from metadata when possible, hardcoded as 70 right now.
