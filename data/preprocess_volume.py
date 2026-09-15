import torch
from torchvision.transforms import Resize

class VolumePreprocessor:
    def __init__(self, preprocess_cfg):
        self.preprocess_cfg = preprocess_cfg


        self.initial_range = preprocess_cfg["INITIAL_RANGE"]
        self.proc_range = preprocess_cfg["PROC_RANGE"]
        self.HU_range = preprocess_cfg.get("HU_RANGE")
        self.mu_wa_mm = preprocess_cfg.get("mu_wa_mm")


        self.d_type = preprocess_cfg["dTYPE"]
        self.shape = preprocess_cfg["SHAPE"]        # Dproc, Hproc, Wproc



    def __call__(self, x):
        x = self.normalize(x)
        x = self.cast(x)
        x = self.reshape(x)
        return x
    
    def normalize(self, volume):
        volume = volume.clone().detach().to(torch.float32)
        volume = (volume - self.initial_range[0]) / (self.initial_range[1] - self.initial_range[0])
        volume = volume * (self.proc_range[1] - self.proc_range[0]) + self.proc_range[0]
        return volume
    

    def rescale_to_HU(self, volume):
        """Rescales from [0, 1] to HU range if HU_RANGE is specified in preprocess_cfg."""
        
        assert self.HU_range is not None, "HU_RANGE is not specified in preprocess_cfg. Please check the preprocess_cfg format."

        volume = volume * (self.HU_range[1] - self.HU_range[0]) + self.HU_range[0]
        return volume

    def rescale_to_attenuation(self, volume):

        """Rescales from HU range to relative attenuation values"""
        
        assert self.mu_wa_mm is not None, "mu_wa_mm is not specified in preprocess_cfg. Please check the preprocess_cfg format."

        volume = volume / 1000.0  + 1 
        return volume



    def cast(self, volume):
        volume = volume.to(getattr(torch, self.d_type))
        return volume
    
    def reshape(self, volume):

        if list(volume.shape[2:]) == self.shape:
            return volume
        else:
            # Reshape volume to the target shape. This is done batch wise, meaning that the first dimension of the volume is the batch dimension and the remaining dimensions are reshaped to the target shape.
            batch_size = volume.shape[0]
            num_time_bins = self.shape[0]

            for i in range(batch_size):
                for t in range(num_time_bins):

                    # TODO make this work for 3D volumes

                    volume[i, t] = Resize(self.shape)(volume[i, t])
            return volume





def preprocess_volume(volume, preprocess_cfg):
    initial_range = preprocess_cfg["INITIAL_RANGE"]
    proc_range = preprocess_cfg["PROC_RANGE"]
    d_type = preprocess_cfg["dTYPE"]
    shape = preprocess_cfg["SHAPE"]

    volume = volume.clone().detach().to(torch.float32)


    volume = (volume - initial_range[0]) / (initial_range[1] - initial_range[0])
    volume = volume * (proc_range[1] - proc_range[0]) + proc_range[0]
    volume = volume.view(*shape)
    volume = volume.to(getattr(torch, d_type))
    return volume