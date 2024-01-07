import os
import torch
import numpy as np
from torch.utils.data import Dataset


class DataSet(Dataset): #定义Dataset类的名称
    def __init__(self,data_path="",volume_shape=(160, 224, 168),mask=False): 
        self.data_path = data_path              # 
        self.volume_shape = volume_shape        # [depth, height, width]
        
        path_list = os.listdir(self.data_path)
        self.path_list = path_list.sort()
        
        self.max_index = ...                    # 最大值
        
    def __getitem__(self, prefix="norm_ct",data_type="raw", index=0):  # index: [0, 69].
        
        assert (index>=0 and index<self.max_index),"The index is out of range"
        assert (not os.path.exists(self.data_path)),"Path is not exist"

        
        # 1. read original volume data.
        fileName = f"{prefix}{index:02d}.{data_type}"
        volume_data = np.fromfile(fileName, dtype=self.float32DataType)
        volume_ct = torch.from_numpy(volume_data)                               # convert numpy data into tensor
        volume_ct = volume_ct.view([1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]])  # reshape into [channels, depth, height, width].


        # 2. given volume_ct/_mr, crop them to get cropped volumes, for data augmentation.
        if self.transform:
            crop_ct, crop_mr = self.transform(volume_ct, volume_mr)

        #make sure crop_ct, crop_mr are the same size.
        assert crop_ct.shape == crop_mr.shape


        return crop_ct, crop_mr, index
        #correct: 2023.5.22.

        
    
    