import os
import random
import torch
import pytest

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset
from enum import Enum

class MaskType(Enum):
    no_mask     = 0
    x_stack     = 1
    y_stack     = 2
    z_stack     = 3
    cuboid      = 4
    cylinder    = 5
    hyperboloid = 6
    sphere      = 7
    tetrahedron = 8
    ring        = 9
    
    
def generate_mask(volume_shape:[int,int,int],shape_type:int):
    '''
        -------------------------
        Input:\n
        shape       -> [int,int,int]     the total size of the dataset\n
        shape_type  -> int               the type of shape \n
            "no_mask"       0
            "x-stack"       1
            "y-stack"       2
            "z-stack"       3
            "cuboid"        4
            "cylinder"      5
            "hyperboloid"   6
            "sphere"        7
            "tetrahedron"   8
            "ring"          9
        -------------------------\n
        Output:\n
        mask        ->  array           mask code with the same size of the "shape"\n
        mask_shape  ->  [int,int,int]   the shape of the shape\n
        mask_pos    ->  [int,int,int]   the position where the mask start
        
    '''
    
    mask = []       # meshgrid vector
    
    # to make sure the biggest size is smaller than 50%
    max_x = volume_shape[0]*0.7
    max_y = volume_shape[1]*0.7
    max_z = volume_shape[2]*0.7
    
    # make empty mask
    mask_volume = np.ones(volume_shape)
    
    if shape_type == 0:         # do nothing
        pass
    elif shape_type == 1:      # x-stack
        # make shape grid
        random_x,random_y,random_z = int(max_x*random.random()),int(max_y*random.random()),int(max_z*random.random())
        x,y,z = np.meshgrid(range(volume_shape[0]),range(random_y),range(random_z))
        x,y,z = np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))
        # make position
        pos_x = random.randint(0,int(volume_shape[0]-random_x))
        pos_y = random.randint(0,int(volume_shape[1]-random_y))
        pos_z = random.randint(0,int(volume_shape[2]-random_z))
        
        # add position
        # mask = [x,y,z]
        mask = [x,y+pos_y,z+pos_z]
        mask_shape = [x.size,y.size,z.size]
        mask_pos = [pos_x,pos_y,pos_z]
        
        for i in range(x.size):
            mask_volume[x[i],y[i],z[i]] = 0

    elif shape_type == 2:    # y-stack
        # make shape grid
        random_x,random_y,random_z = int(max_x*random.random()),int(max_y*random.random()),int(max_z*random.random())
        x,y,z = np.meshgrid(range(random_x),range(volume_shape[1]),range(random_z))
        x,y,z = np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))
        # make position
        pos_x = random.randint(0,int(volume_shape[0]-random_x))
        pos_y = random.randint(0,int(volume_shape[1]-random_y))
        pos_z = random.randint(0,int(volume_shape[2]-random_z))
        
        # add position
        # mask = [x,y,z]
        mask = [x+pos_x,y,z+pos_z]
        mask_shape = [x.size,y.size,z.size]
        mask_pos = [pos_x,pos_y,pos_z]
        
        for i in range(x.size):
            mask_volume[x[i],y[i],z[i]] = 0
        
    elif shape_type == 3:    # z-stack
        # make shape grid
        random_x,random_y,random_z = int(max_x*random.random()),int(max_y*random.random()),int(max_z*random.random())
        x,y,z = np.meshgrid(range(random_x),range(random_y),range(volume_shape[2]))
        x,y,z = np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))
        # make position
        pos_x = random.randint(0,int(volume_shape[0]-random_x))
        pos_y = random.randint(0,int(volume_shape[1]-random_y))
        pos_z = random.randint(0,int(volume_shape[2]-random_z))
        
        # add position
        # mask = [x,y,z]
        mask = [x+pos_x,y+pos_y,z]
        mask_shape = [x.size,y.size,z.size]
        mask_pos = [pos_x,pos_y,pos_z]
        
        for i in range(x.size):
            mask_volume[x[i],y[i],z[i]] = 0
        
    elif shape_type == 4:    # cuboid
        # make shape grid
        random_x,random_y,random_z = int(max_x*random.random()),int(max_y*random.random()),int(max_z*random.random())
        x,y,z = np.meshgrid(range(random_x),range(random_y),range(random_z))
        x,y,z = np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))
        # make position
        pos_x = random.randint(0,int(volume_shape[0]-random_x))
        pos_y = random.randint(0,int(volume_shape[1]-random_y))
        pos_z = random.randint(0,int(volume_shape[2]-random_z))
        
        # add position
        # mask = [x,y,z]
        mask = [x+pos_x,y+pos_y,z+pos_z]
        mask_shape = [x.size,y.size,z.size]
        mask_pos = [pos_x,pos_y,pos_z]
        
        for i in range(x.size):
            mask_volume[x[i],y[i],z[i]] = 0
        
    elif shape_type == 5:    # cylinder         圆柱体
        # make shape grid
        random_x,random_y,random_z = int(max_x*random.random()),int(max_y*random.random()),int(max_z*random.random())
        x,y,z = np.meshgrid(range(random_x),range(random_y),range(random_z))
        x,y,z = np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))
        # make position
        pos_x = random.randint(0,int(volume_shape[0]-random_x))
        pos_y = random.randint(0,int(volume_shape[1]-random_y))
        pos_z = random.randint(0,int(volume_shape[2]-random_z))
        
        # add position
        # mask = [x,y,z]
        mask = [x+pos_x,y+pos_y,z+pos_z]
        mask_shape = [x.size,y.size,z.size]
        mask_pos = [pos_x,pos_y,pos_z]
        
        for i in range(x.size):
            mask_volume[x[i],y[i],z[i]] = 0
        
        
        
        
        r = 1
        h = 2
        theta = np.linspace(0, 2 * np.pi)
        
        
        
        
        
    elif shape_type == 6:    # hyperboloid      双曲线体
        ...
    elif shape_type == 7:    # sphere           球体
        ...
    elif shape_type == 8:    # tetrahedron      四面体
        ...
    elif shape_type == 9:    # ring             环
        ...
    else:
        assert True,"wrong input parameter"
    
    # mask_volume = mask_volume.view([1, volume_shape[0], volume_shape[1], volume_shape[2]])   # reshape into [channels, depth, height, width].
    
    return mask_volume,mask
    # return mask_volume,mask


class DataSet(Dataset): #定义Dataset类的名称
    def __init__(self,data_path="", volume_shape=(160, 224, 168), mask_type="test", prefix="norm_ct.",data_type="raw",max_index=70): 
        self.data_path = data_path              
        self.prefix = prefix
        self.max_index = max_index                   # largest index
        self.data_type = data_type
        self.volume_shape = volume_shape        # [depth, height, width]
        
        self.mask_type = mask_type              # "test","train","predict"
        self.mask_name = ["no_mask","x_stack","y_stack","z_stack","cuboid","cylinder","hyperboloid","sphere","tetrahedron","ring"]
        
        path_list = os.listdir(self.data_path)
        self.path_list = path_list.sort()
    
    def __len__(self):
        return self.max_index     # =70.
    
        
    def __getitem__(self, index=0):               # index: [0, 69].

        assert (index<0 or index>=self.max_index),f"The index {index} of the data is out of range"
        assert (not os.path.exists(self.data_path)),f"Path [{self.data_path}] is not exist"

        # read original volume data.
        fileName = f"{self.prefix}{index:02d}.{self.data_type}"
        volume_data = np.fromfile(fileName, dtype=self.float32DataType)
        volume_data = torch.from_numpy(volume_data)                                                             # convert numpy data into tensor
        
        # generate mask (if needed)
        if self.mask_type == "test":
            # make empty mask               1:no mask  0:mask
            masked_volume_data = torch.from_numpy(np.ones(self.volume_shape))
            masked_volume_data = masked_volume_data.view([1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]])   # reshape into [channels, depth, height, width].
        
        elif (self.mask_type == "train"):
            # mask when train
            mask_index = random.randint(1,3)              #123
            mask_name = self.mask_name[mask_index]
            mask_volume,mask = generate_mask(volume_shape=volume_data.shape, shape_type=mask_index)
            masked_volume_data = volume_data * mask_volume
            
        elif (self.mask_type == "predict"):
            # mask when predict
            mask_index = random.randint(3,9)              #3456789
            mask_name = self.mask_name[mask_index]
            mask_volume,mask = generate_mask(volume_shape=volume_data.shape, shape_type=mask_index)
            masked_volume_data = volume_data * mask_volume
            
                
        # mask_volume = torch.from_numpy(mask_volume)
        volume_data = volume_data.view([1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]])   # reshape into [channels, depth, height, width].
        
        
        return volume_data,masked_volume_data,index

        
    
    