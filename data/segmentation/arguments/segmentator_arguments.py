from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SegmentationArguments:

    dir: str = field(default='/mnt/data/trulssv/LDDMM/processed',
                     metadata= {"help": "The main directory of the dataset"})
    cases: str = field(default='0:1',
                       metadata={"help": 'Slice or list of cases to process, e.g. "0:10" or "1,2,3"'})
    license_number: str = field(default='aca_E405RX375O77FR',
                               metadata={"help": "The license number"})
    mode: bool = field(default=False,
                      metadata={"help": "Use fast mode (model trained on 3 mm slices) for segmentation. If false, the fine model trained on 1 mm slices will be used"})
    tasks: Tuple[str] = field(default = ("lung_nodules",))
    