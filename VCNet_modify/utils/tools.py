import os
import random
import torch
# import pytest

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset
from enum import Enum

# saveRawFile10 --> not test assume good    --2024.1.22
def saveRawFile10(dataSavePath, fileName, volume):
    if not os.path.exists(dataSavePath):
        os.makedirs(dataSavePath)
    fileName = f"{dataSavePath}/{fileName}.raw"

    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)
    
# def detect_mask(real_volume,masked_volume)


class DataSet(Dataset):
    # brain: [depth, height, width]. dim = (160, 224, 168)   
    def __init__(self,data_path="",volume_shape=[128,128,128],target_shape=[128,128,128],mask_type="train",data_type=np.float32):
        self.data_path = data_path
        self.volume_shape = volume_shape
        self.target_shape = target_shape
        self.mask_type = mask_type
        self.data_type = data_type
        self.volumes = os.listdir(data_path)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx=0): 
        assert(self.mask_type=="train" or self.mask_type=="test"),"input 'train' or 'test' as mask type !"
        file_name = os.path.join(self.data_path, self.volumes[idx])
        data = np.fromfile(file_name, dtype=self.data_type)
        data.resize(self.volume_shape)
        
        # resize to target shape
        if np.prod(data.shape)<np.prod(self.target_shape):
            print("input size too small!!")
        elif np.prod(data.shape)>np.prod(self.target_shape):
            print("crop!")
            data = self.crop(data,self.target_shape)
        
        #generate mask
        shape_type = random.randint(1,4) if self.mask_type=="train" else random.randint(4,9)
        print(shape_type)
        mask = self.generate_mask(self.target_shape,shape_type=4)
        
        return data,mask
        
        
    def crop(self,data, new_shape):
        '''
        Function for cropping an image tensor: Given an image tensor and the new shape,
        crops to the center pixels (assumes that the input's size and the new size are
        even numbers).
        Parameters:
            image: image tensor of shape (batch size, channels, height, width)
            new_shape: a torch.Size object with the shape you want x to have
        '''
        middle_depth=data.shape[0] //2
        middle_height = data.shape[1] // 2
        middle_width = data.shape[2] // 2
        starting_depth=middle_depth-new_shape[0]//2
        final_depth=starting_depth+new_shape[0]
        starting_height = middle_height - new_shape[1] // 2
        final_height = starting_height + new_shape[1]
        starting_width = middle_width - new_shape[2] // 2
        final_width = starting_width + new_shape[2]
        cropped_image = data[starting_depth:final_depth, starting_height:final_height, starting_width:final_width]
        return cropped_image

    def generate_mask(self,volume_shape:tuple[int,int,int],shape_type:int):
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
        max_x = volume_shape[0]*0.3
        max_y = volume_shape[1]*0.3
        max_z = volume_shape[2]*0.3
        
        # make empty mask
        mask_volume = np.zeros(volume_shape)
        
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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 1

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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 1
            
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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 1

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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 1
            
        elif shape_type == 5:    # cylinder         圆柱体
            random_r,random_h = int(volume_shape[0]*random.random()),int(volume_shape[2]*random.random())
            
            # make sure the size is less than 1/2 origin
            while((random_r**2) * np.pi * random_h > volume_shape[0]*volume_shape[1]*volume_shape[2]/2):
                random_r,random_h = int(volume_shape[0]*random.random()),int(volume_shape[2]*random.random())
                
            # make position
            pos_x = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            pos_y = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            pos_z = random.randint(0,int(volume_shape[2]-random_h))
            
            # calculate the distance, if distance_xy<r and z in the h, set 1
            for i in range(volume_shape[0]):
                for j in range(volume_shape[1]):
                    for k in range(volume_shape[2]):
                        if ((i-pos_x)**2 + (j-pos_y)**2 <= random_r**2)and(k>=pos_z and k<=pos_z+random_h):
                            mask_volume[i,j,k] = 1
            
            mask = "cylinder"
            
        elif shape_type == 6:    # hyperboloid      双曲线体 暂时用球体替代
            random_r = int(volume_shape[0]*random.random())
            
            # make sure the size is less than 1/2 origin
            while((random_r**2) * np.pi * 4/3 > volume_shape[0]*volume_shape[1]*volume_shape[2]/2):
                random_r = int(volume_shape[0]*random.random())
                
            # make position
            random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r)
            pos_x = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            pos_y = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            pos_z = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            
            # calculate the distance, if distance_xy<r and z in the h, set 1
            for i in range(volume_shape[0]):
                for j in range(volume_shape[1]):
                    for k in range(volume_shape[2]):
                        if ((i-pos_x)**2 + (j-pos_y)**2 + (k-pos_z)**2 <= random_r**2):
                            mask_volume[i,j,k] = 1
                            
        elif shape_type == 7:    # sphere           球体
            random_r = int(volume_shape[0]*random.random())
            
            # make sure the size is less than 1/2 origin
            while((random_r**2) * np.pi * 4/3 > volume_shape[0]*volume_shape[1]*volume_shape[2]/2):
                random_r = int(volume_shape[0]*random.random())
                
            # make position
            random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r)
            pos_x = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            pos_y = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            pos_z = random.randint(random_r if (random_r<=int(volume_shape[0]-random_r)) else int(volume_shape[0]-random_r),
                                int(volume_shape[0]-random_r) if (random_r>int(volume_shape[0]-random_r)) else random_r)
            
            # calculate the distance, if distance_xy<r and z in the h, set 1
            for i in range(volume_shape[0]):
                for j in range(volume_shape[1]):
                    for k in range(volume_shape[2]):
                        if ((i-pos_x)**2 + (j-pos_y)**2 + (k-pos_z)**2 <= random_r**2):
                            mask_volume[i,j,k] = 1
                            
        elif shape_type == 8:    # tetrahedron      四面体
            ...
        elif shape_type == 9:    # ring             环
            ...
        else:
            assert True,"wrong input parameter"
        
        # mask_volume = mask_volume.view([1, volume_shape[0], volume_shape[1], volume_shape[2]])   # reshape into [channels, depth, height, width].
        # mask_volume = np.array(mask)
        return mask_volume


def test_dataset():
    dataset = DataSet(data_path="/Users/wanglikai/Codes/DataSets/dataSet0",
                      volume_shape=(160,224,168),
                      target_shape=(128,128,128),
                      mask_type="train")
    data,mask = dataset.__getitem__(1)
    print(data.shape)
    print(mask.shape)


if __name__ == '__main__':
    test_dataset()
