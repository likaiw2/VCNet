import os
import random
import torch
import pytest

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def saveRawFile10(dataSavePath, fileName, volume):
    if not os.path.exists(dataSavePath):
        os.makedirs(dataSavePath)
    fileName = f"{dataSavePath}/{fileName}.raw"

    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)

    
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
            mask_volume[x[i],y[i],z[i]] = 1

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
            mask_volume[x[i],y[i],z[i]] = 1
        
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
            mask_volume[x[i],y[i],z[i]] = 1
        
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
            mask_volume[x[i],y[i],z[i]] = 1
        
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
            mask_volume[x[i],y[i],z[i]] = 1
        
        
        
        
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
            # make empty mask               1:mask  0:no mask
            masked_volume_data = torch.from_numpy(np.ones(self.volume_shape))
            masked_volume_data = masked_volume_data.view([1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]])   # reshape into [channels, depth, height, width].
        
        elif (self.mask_type == "train"):
            # mask when train
            mask_index = random.randint(1,3)              #123
            mask_name = self.mask_name[mask_index]
            mask_volume,mask = generate_mask(volume_shape=volume_data.shape, shape_type=mask_index)
            masked_volume_data = volume_data * (1 - mask_volume)
            
        elif (self.mask_type == "predict"):
            # mask when predict
            mask_index = random.randint(3,9)              #3456789
            mask_name = self.mask_name[mask_index]
            mask_volume,mask = generate_mask(volume_shape=volume_data.shape, shape_type=mask_index)
            masked_volume_data = volume_data * (1 - mask_volume)
            
                
        # mask_volume = torch.from_numpy(mask_volume)
        volume_data = volume_data.view([1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]])   # reshape into [channels, depth, height, width].
        
        
        return volume_data,masked_volume_data,mask_volume,index


# LOSS Function Class
class WeightedMSELoss(nn.Module):
    # $\mathcal{L}_{\mathrm{rec}}^G=\frac{1}{n} \sum_{j=1}^n\left\|\mathbf{M}_j^C \odot\left(G\left(\mathbf{V}_{M, j}^C\right)-\mathbf{V}_j^C\right)\right\|_2$
    
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        ones = torch.ones_like(y_true[0, :])
        idx = torch.cumsum(ones, dim=0)             # calculate the accumulate sum
        weights = 1 / idx

        mse_loss = F.mse_loss(y_true, y_pred, reduction='none')
        weighted_mse_loss = torch.mean(weights * mse_loss)

        return weighted_mse_loss

class AdversarialGLoss(nn.Module):
    # $\mathcal{L}_{\mathrm{adv}}^G=\frac{1}{n} \sum_{j=1}^n\left[\log D\left(\mathbf{M}_j^C \odot G\left(\mathbf{V}_{M, j}^C\right)+\left(\mathbf{1}-\mathbf{M}_j^C\right) \odot \mathbf{V}_j^C\right)\right]$
    def __init__(self, discriminator):
        super(AdversarialGLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, G_output, V_M, M, V_real):
        # G_output: Generator output for input V_M
        # V_M: Masked input for generator
        # M: Mask matrix
        # V_real: Real data
        
        # Calculate the generated data mixed with real data
        mixed_data = M * G_output + (1 - M) * V_real

        # Calculate the discriminator's predictions for the mixed data
        discriminator_output = self.discriminator(mixed_data)

        # Calculate the adversarial loss using binary cross entropy
        adversarial_loss = F.binary_cross_entropy_with_logits(discriminator_output, torch.ones_like(discriminator_output))
        
        # Calculate the average adversarial loss across samples
        n = G_output.size(0)  # Assuming the batch size is the first dimension
        adversarial_loss = adversarial_loss / n

        return adversarial_loss

class AdversarialDLoss(nn.Module):
    def __init__(self, discriminator):
        super(AdversarialDLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, V_real, G_output, V_M, M):
        # V_real: Real data
        # G_output: Generator output for input V_M
        # V_M: Masked input for generator
        # M: Mask matrix

        # Calculate the discriminator's predictions for real data
        real_discriminator_output = self.discriminator(V_real)

        # Calculate the adversarial loss for real data
        loss_real = F.binary_cross_entropy_with_logits(real_discriminator_output, torch.ones_like(real_discriminator_output))

        # Calculate the generated data mixed with real data
        mixed_data = M * G_output + (1 - M) * V_real

        # Calculate the discriminator's predictions for mixed data
        mixed_discriminator_output = self.discriminator(mixed_data)

        # Calculate the adversarial loss for generated data
        loss_generated = F.binary_cross_entropy_with_logits(mixed_discriminator_output, torch.zeros_like(mixed_discriminator_output))

        # Calculate the average adversarial loss across samples
        n = V_real.size(0)  # Assuming the batch size is the first dimension
        total_loss = (loss_real + loss_generated) / (2 * n)

        return total_loss







