
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.utils import make_grid
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import math
import time
import datetime
import matplotlib.pyplot as plt
import os
import random

torch.manual_seed(0)
# modified RandomCrop3D class (refer to: https://discuss.pytorch.org/t/efficient-way-to-crop-3d-image-in-pytorch/78421), which:
# with one call, crop a pair of (volume_ct/volume_mr) at the same position;
# with different calls, randomly crop volumes at different positions.
def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels (assumes that the input's size and the new size are
    even numbers).
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    middle_depth=image.shape[2] //2
    middle_height = image.shape[3] // 2
    middle_width = image.shape[4] // 2
    starting_depth=middle_depth-new_shape[2]//2
    final_depth=starting_depth+new_shape[2]
    starting_height = middle_height - new_shape[3] // 2
    final_height = starting_height + new_shape[3]
    starting_width = middle_width - new_shape[4] // 2
    final_width = starting_width + new_shape[4]
    cropped_image = image[:, :,starting_depth:final_depth, starting_height:final_height, starting_width:final_width]
    return cropped_image

def saveRawFile10(cur_step, res, t, l, volume):
    fileName = '%s%s_%.4d_%s_%d.raw' % (dataSavePath, res, t, l, cur_step)
    volume = volume.view(dim_crop[0], dim_crop[1], dim_crop[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)
# add on 2022.9.26.


def saveRawFile2(cur_step, t, volume):
    fileName = '%sH_%s_%d.raw' % (dataSavePath, t, cur_step)
    volume = volume.view(dim_crop[0], dim_crop[1], dim_crop[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)

def saveModel(cur_step):
    if save_model:
        fileName = "%spix2pix_%d.pth" % (dataSavePath, cur_step)
        torch.save({'gen': gen.state_dict(),
                    'gen_opt': gen_opt.state_dict(),
                    # 'disc_S': disc_S.state_dict(),
                    # 'disc_T': disc_T.state_dict(),
                    # 'disc_S_opt': disc_S_opt.state_dict(),
                    # 'disc_T_opt': disc_T_opt.state_dict(),
                    }, fileName)  # , _use_new_zipfile_serialization=False)
        
# class MyRandomCrop3D3(object):
#     def __init__(self, volume_sz, cropVolume_sz):
#         c, d, h, w = volume_sz
#         assert (d, h, w) >= cropVolume_sz
#         self.volume_sz = tuple((d, h, w))
#         self.cropVolume_sz = tuple(cropVolume_sz)

#     def __call__(self, volume_ct, volume_mr):
#         slice_dhw = [self._get_slice(i, k) for i, k in zip(self.volume_sz, self.cropVolume_sz)]
#         return self._crop(volume_ct, volume_mr, *slice_dhw)

#     @staticmethod
#     def _get_slice(volume_sz, cropVolume_sz):
#         try:
#             lower_bound = torch.randint(volume_sz - cropVolume_sz, (1,)).item()
#             return lower_bound, lower_bound + cropVolume_sz
#         except:
#             return (None, None)

#     @staticmethod
#     def _crop(volume_ct, volume_mr, slice_d, slice_h, slice_w):
#         # print(f"slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]: {slice_d[0], slice_d[1], slice_h[0], slice_h[1], slice_w[0], slice_w[1]}")
#         return volume_ct[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
#                volume_mr[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]

class MyRandomCrop3D3(object):
    # 这个类，用于实现随机裁剪
    def __init__(self, volume_sz, cropVolume_sz):
        d, h, w = volume_sz                  # 输入体积大小
        # assert (d, h, w) >= cropVolume_sz
        self.volume_sz = tuple((d, h, w))       #裁剪体积大小
        self.cropVolume_sz = tuple(cropVolume_sz)

    def __call__(self, volume_ct):
        slice_dhw = [self._get_slice(i, k) for i, k in zip(self.volume_sz, self.cropVolume_sz)]     #记录裁剪的大小和位置
        return self._crop(volume_ct, *slice_dhw)

    @staticmethod
    def _get_slice(volume_sz, cropVolume_sz):           # 随机生成所需裁剪的切片位置，并处理边界情况，确保切片范围在给定的 3D 体积数组内或返回 None来处理捕捉到的异常
        try:
            lower_bound = torch.randint(volume_sz - cropVolume_sz, (1,)).item()
            return lower_bound, lower_bound + cropVolume_sz
        except:
            return (None, None)

    # 返回裁切后的数据
    @staticmethod
    def _crop(volume_ct, slice_d, slice_h, slice_w):     
        return volume_ct[slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]

class VolumesDataset(Dataset):
    def __init__(self, dataSourcePath, nTimesteps_train, dim,
                 fileStartVal, fileIncrement, constVal, float32DataType=np.float32,
                 transform=None):
        self.dataSourcePath = dataSourcePath
        self.nTimesteps_train = nTimesteps_train  # number of timesteps used for training.
        self.fileStartVal = fileStartVal
        self.fileIncrement = fileIncrement
        self.constVal = constVal
        self.float32DataType = float32DataType
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return self.nTimesteps_train  # =70.

    # given an index, return a pair of (ct, mr).
    def __getitem__(self, index):  # index: [0, 69].
        # if index is outside normal range.
        if index < 0 or index >= self.nTimesteps_train:
            print('index is outside the normal range.\n')
            return


        # 1. at index, read original a pair of (volume_ct, volume_mr) .raw files.
        # (1.1)read original volume_ct.
        fileName = '%snorm_ct_enContrast.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_ct = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_ct = torch.from_numpy(volume_ct)
        # reshape.
        volume_ct = volume_ct.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # (1.2)read original volume_mr.
        fileName = '%snorm_mr_enContrast.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_mr = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_mr = torch.from_numpy(volume_mr)
        # reshape.
        volume_mr = volume_mr.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].


        # 2. given volume_ct/_mr, crop them to get cropped volumes, for data augmentation.
        if self.transform:
            crop_ct, crop_mr = self.transform(volume_ct, volume_mr)

        #make sure crop_ct, crop_mr are the same size.
        assert crop_ct.shape == crop_mr.shape


        return crop_ct, crop_mr, index
        #correct: 2023.5.22.


class DataSet(Dataset):
    # brain: [depth, height, width]. dim = (160, 224, 168)   
    def __init__(self,data_path="",volume_shape=[128,128,128],target_shape=[128,128,128],mask_type="train",data_type=np.float32,transform=None):
        self.data_path = data_path
        self.volume_shape = volume_shape
        self.target_shape = target_shape
        self.mask_type = mask_type
        self.data_type = data_type
        self.volumes = os.listdir(data_path)
        self.transform = MyRandomCrop3D3(volume_sz = self.volume_shape,
                                         cropVolume_sz=self.target_shape)
        
    def __len__(self):
        return len(self.volumes)
        
    def __getitem__(self, idx=0): 
        assert(self.mask_type=="train" or self.mask_type=="test"),"input 'train' or 'test' as mask type !"
        file_name = os.path.join(self.data_path, self.volumes[idx])
        data = np.fromfile(file_name, dtype=self.data_type)
        data.resize(self.volume_shape)
        # print(data.dtype)
        
        # resize to target shape
        if np.prod(data.shape)<np.prod(self.target_shape):
            # print("input size too small!!")
            pass
        elif np.prod(data.shape)>np.prod(self.target_shape):
            # print("crop!")
            data = self.transform(data)
        
        #generate mask
        shape_type = random.randint(1,4) if self.mask_type=="train" else random.randint(4,9)
        # print(shape_type)
        mask = self.generate_mask(self.target_shape,shape_type=4)
        
        data = data.reshape([1,self.target_shape[0],self.target_shape[1],self.target_shape[2]])
        mask = mask.reshape([1,self.target_shape[0],self.target_shape[1],self.target_shape[2]])

        return data,mask*data,idx

    def crop(self,data, new_shape):
        '''
        Function for cropping an image tensor: Given an image tensor and the new shape,
        crops to the center pixels (assumes that the input's size and the new size are
        even numbers).
        Parameters:
            image: image tensor of shape (batch size, channels, height, width)
            new_shape: a torch.Size object with the shape you want x to have
        '''
        middle_depth = data.shape[0] // 2
        middle_height = data.shape[1] // 2
        middle_width = data.shape[2] // 2
        
        starting_depth=middle_depth-new_shape[0]//2
        final_depth=starting_depth+new_shape[0]
        
        starting_height = middle_height - new_shape[1] // 2
        final_height = starting_height + new_shape[1]
        
        starting_width = middle_width - new_shape[2] // 2
        final_width = starting_width + new_shape[2]
        
        cropped_image = data[starting_depth:final_depth, 
                             starting_height:final_height, 
                             starting_width:final_width]
        return cropped_image

    def generate_mask(self,volume_shape:[int,int,int],shape_type:int):
        '''
            1 stands for real
            0 stands for empty
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
        max_x = int(volume_shape[0]*0.7)
        max_y = int(volume_shape[1]*0.7)
        max_z = int(volume_shape[2]*0.7)
        min_x = int(volume_shape[0]*0.3)
        min_y = int(volume_shape[0]*0.3)
        min_z = int(volume_shape[0]*0.3)
        
        # make empty mask
        mask_volume = np.ones(volume_shape,dtype=self.data_type)
        
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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 0

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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 0
            
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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 0

        elif shape_type == 4:    # cuboid
            # make shape grid
            random_x = random.randint(min_x,max_x)
            random_y = random.randint(min_y,max_y)
            random_z = random.randint(min_z,max_z)
            
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
                mask_volume[mask[0][i],mask[1][i],mask[2][i]] = 0
            
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
                            mask_volume[i,j,k] = 0
            
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
                            mask_volume[i,j,k] = 0
                            
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
                            mask_volume[i,j,k] = 0
                            
        elif shape_type == 8:    # tetrahedron      四面体
            ...
        elif shape_type == 9:    # ring             环
            ...
        else:
            assert True,"wrong input parameter"
        
        # mask_volume = mask_volume.view([1, volume_shape[0], volume_shape[1], volume_shape[2]])   # reshape into [channels, depth, height, width].
        # mask_volume = np.array(mask)
        return mask_volume


# def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
#     '''
#     Function for visualizing images: Given a tensor of images, number of images, and
#     size per image, plots and prints the images in an uniform grid.
#     '''
#     image_shifted = image_tensor
#     image_unflat = image_shifted.detach().cpu().view(-1, *size)
#     image_grid = make_grid(image_unflat[:num_images], nrow=5)
#     plt.imshow(image_grid.permute(1, 2, 0).squeeze())
#     plt.show()

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def     __init__(self, input_channels, use_dropout=False, use_bn=True,s1=2,k1=2):
        super(ContractingBlock, self).__init__()
        #self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool3d(kernel_size=k1, stride=s1)
        if use_bn:
            #self.batchnorm = nn.BatchNorm2d(input_channels * 2)
            # self.batchnorm = nn.BatchNorm3d(input_channels * 2)
            self.instancenorm=nn.InstanceNorm3d(input_channels*2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True,s1=2):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=s1, mode='trilinear', align_corners=True)
        # self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        # self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        self.conv1 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            # self.batchnorm = nn.BatchNorm2d(input_channels // 2)
            self.batchnorm = nn.BatchNorm3d(input_channels // 2)
            self.instancenorm=nn.InstanceNorm3d(input_channels//2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        # print(x.shape)
        #[2048,1,1,1]

        x = self.upsample(x)
        # print("After Expanding:"+str(x.shape))
        # print(x.shape)
        x = self.conv1(x)
        # print("After Conv1:" + str(x.shape))
        # print(x.shape)
        #使用裁剪操作将上一步生成的数据裁剪过后与现有生成数据拼接
        skip_con_x = crop(skip_con_x, x.shape)
        # print("skip_con_x.shape:"+str(skip_con_x.shape))
        # print("x.shape:" + str(x.shape))
        x = torch.cat([x, skip_con_x], axis=1)
        # print("After concact:"+x.shape)
        x = self.conv2(x)
        # print("X AFTER CONV2:"+str(x.shape))
        if self.use_bn and x.shape[3] >1 :
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn :
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net -
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        # self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class UNet(nn.Module):
    #Generator
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        # self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True,s1=(2,2,3))
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        # self.contract4 = ContractingBlock(hidden_channels * 8,k1=(2,2,1),s1=(2,2,1))
        self.contract4 = ContractingBlock(hidden_channels * 8,k1=(1,2,1),s1=(1,2,1))
        # self.contract5 = ContractingBlock(hidden_channels * 16,k1=(1,1,1),s1=(1,1,1))
        self.contract5 = ContractingBlock(hidden_channels * 16, k1=(1,1,1),s1=(1,1,1))
        self.contract6 = ContractingBlock(hidden_channels * 32,k1=(1,1,1),s1=(1,1,1))
        self.expand0 = ExpandingBlock(hidden_channels * 64,s1=1)
        self.expand1 = ExpandingBlock(hidden_channels * 32,s1=1)
        # self.expand1 = ExpandingBlock(hidden_channels * 32, s1=(1,2,1))
        # self.expand2 = ExpandingBlock(hidden_channels * 16,s1=(2,2,1))
        self.expand2 = ExpandingBlock(hidden_channels * 16,s1=(1,2,1))
        # self.expand3 = ExpandingBlock(hidden_channels * 8,s1=(2,2,3))
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        # self.deconv=nn.ConvTranspose3d(1,1,kernel_size=(9,9,6),stride=1,padding=0,dilation=(2,6,4))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet:
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        # print("X3.shape:"+str(x3.shape))
        x4 = self.contract4(x3)
        # print("X4.shape:" + str(x4.shape))
        x5 = self.contract5(x4)
        # print("X5.shape:" + str(x5.shape))
        x6 = self.contract6(x5)
        # print("X6.shape:" + str(x6.shape))

        # print("X6.shape:"+str(x6.shape))
        # print(x5.shape)
        # x6=x6.squeeze(0)
        x7 = self.expand0(x6, x5)
        # print("X7.shape:" + str(x7.shape))
        x8 = self.expand1(x7, x4)
        # print("X8.shape:" + str(x8.shape))
        x9 = self.expand2(x8, x3)
        # print("X9.shape:" + str(x9.shape))
        x10 = self.expand3(x9, x2)
        # print("X10.shape:" + str(x10.shape))
        x11 = self.expand4(x10, x1)
        # print("X11.shape:" + str(x11.shape))
        x12 = self.expand5(x11, x0)
        # print("X12.shape:" + str(x12.shape))
        # print("X12 shape"+str(x12.shape))
        xn = self.downfeature(x12)
        # print("Xn shape" + str(xn.shape))
        # xn=self.deconv(xn)

        return self.sigmoid(xn)

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        #### START CODE HERE ####
        # self.final = nn.Conv2d(hidden_channels * 16, None, kernel_size=None)
        self.final = nn.Conv3d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x, y):
        # print(x.shape)
        # print(y.shape)
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn

# # UNIT TEST
# test_discriminator = Discriminator(10, 1)
# assert tuple(test_discriminator(
#     torch.randn(1, 5, 256, 256),
#     torch.randn(1, 5, 256, 256)
# ).shape) == (1, 1, 16, 16)
# print("Success!")

#Training Process
import torch.nn.functional as F
# New parameters
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200
device=torch.device("cuda:0")
n_epochs = 1000#1500
input_dim = 1
real_dim = 1
display_step = 200
batch_size = 1
lr = 0.0002
target_shape = 256
# device = 'cuda'
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
totalTimesteps = 100
trainsetRatio = 0.7  # according to deep learning specialization, if you have a small collection of data, then use 70%/30% for train/test.
nTimesteps_train = round(totalTimesteps * trainsetRatio)  # nTimesteps_train=70.
# dataSourcePath = '/home/dell/storage/XIEYETAO/preprocess_globalNormAndEnContrast_pelvis/'
# dataSavePath = '/home/dell/storage/XIEYETAO/crop=0.5/epoch=1500_lr=0.0002_instancenorm_pelvis/'
dataSourcePath = "/root/autodl-tmp/Diode/Datas/VCNet_dataSet"
dataSavePath = "/root/autodl-tmp/Diode/Codes/Volume_Impainting/pix2pix_m/out/"

fileStartVal = 1
fileIncrement = 1
constVal = 1
cropScaleFactor = (0.5, 0.5, 0.5)  # [depth, height, width].
# dim = (160, 224, 168)  # [depth, height, width].
# dim = (96, 384, 240)  # [depth, height, width].#for pelvis
dim = (128, 128, 128)  # [depth, height, width].#for inpainting

# dim_crop = (int(dim[0] * cropScaleFactor[0]),
#          int(dim[1] * cropScaleFactor[1]),
#          int(dim[2] * cropScaleFactor[2]))
dim_crop = dim

float32DataType = np.float32
# myRandCrop3D = MyRandomCrop3D3(volume_sz=(1, dim[0], dim[1], dim[2]),
#                                cropVolume_sz=dim_crop)
import torchvision
# dataset = torchvision.datasets.ImageFolder("maps", transform=transform)
# trainDataset=VolumesDataset(dataSourcePath=dataSourcePath, nTimesteps_train=nTimesteps_train,
#                               dim=dim,
#                               fileStartVal=fileStartVal, fileIncrement=fileIncrement, constVal=constVal,
#                               float32DataType=float32DataType,
#                               transform=myRandCrop3D)

trainDataset=DataSet(data_path=dataSourcePath,volume_shape=[128,128,128],target_shape=dim_crop)

gen = UNet(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Feel free to change pretrained to False if you're training the model from scratch
pretrained = False
save_model=True

if pretrained:
    loaded_state = torch.load("pix2pix_15000.pth")
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    disc.load_state_dict(loaded_state["disc"])
    disc_opt.load_state_dict(loaded_state["disc_opt"])
else:
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: get_gen_loss
def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator
                  predictions and the true labels and returns a adversarial
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator
                    outputs and the real images and returns a reconstructuion
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''
    # Steps: 1) Generate the fake images, based on the conditions.
    #        2) Evaluate the fake images and the condition with the discriminator.
    #        3) Calculate the adversarial and reconstruction losses.
    #        4) Add the two losses, weighting the reconstruction loss appropriately.
    #### START CODE HERE ####
    fake = gen(condition)
    disc_fake_hat = disc(fake, condition)
    gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    gen_rec_loss = recon_criterion(real, fake)
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
    #### END CODE HERE ####
    return gen_loss
# UNIT TEST
def test_gen_reasonable(num_images=10):
    gen = torch.zeros_like
    disc = lambda x, y: torch.ones(len(x), 1)
    real = None
    condition = torch.ones(num_images, 3, 10, 10)
    adv_criterion = torch.mul
    recon_criterion = lambda x, y: torch.tensor(0)
    lambda_recon = 0
    assert get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon).sum() == num_images

    disc = lambda x, y: torch.zeros(len(x), 1)
    assert torch.abs(get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)).sum() == 0

    adv_criterion = lambda x, y: torch.tensor(0)
    recon_criterion = lambda x, y: torch.abs(x - y).max()
    real = torch.randn(num_images, 3, 10, 10)
    lambda_recon = 2
    gen = lambda x: real + 1
    assert torch.abs(get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon) - 2) < 1e-4

    adv_criterion = lambda x, y: (x + y).max() + x.max()
    assert torch.abs(get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon) - 3) < 1e-4
test_gen_reasonable()
print("Success!")
# from skimage import color
import numpy as np

def train(save_model=True):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    start_time = time.time()
    ot = time.time()
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("##train start(brain)##  time:",t1)
    
    for epoch in tqdm(range(n_epochs)):
        # Dataloader returns the batches
        for data,masked_data,index in dataloader:
            ct=masked_data
            mri=data
            cur_batch_size=len(ct)
            ct=ct.to(device)
            mri=mri.to(device)
            # print("Ct's shape:"+str(ct.shape))
            # print("MRI's shape:"+str(mri.shape))
            # image_width = image.shape[3]
            #condition代表真实ct图像
            #real代表真实mri图像
            #fake代表生成器生成的假mri图像
            # condition = image[:, :, :, :image_width // 2]
            # condition = nn.functional.interpolate(condition, size=target_shape)
            # real = image[:, :, :, image_width // 2:]
            # real = nn.functional.interpolate(real, size=target_shape)
            # cur_batch_size = len(condition)
            # condition = condition.to(device)
            # real = real.to(device)

            ### Update discriminator ###
            disc_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake = gen(ct)
                # print("Fake shape:"+str(fake.shape))
            disc_fake_hat = disc(fake.detach(), ct) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(mri, ct)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, mri, ct, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###

            
            if (cur_step+1) % display_step == 0 or cur_step == 1:
                
                t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                print(t, f"\n    Epoch {epoch}: Step {cur_step}: Generator (Res_UNet) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                #计算单位运行时间
                dt = time.time() - ot
                elapsedTime = str(datetime.timedelta(seconds=dt))
                per_epoch = str(datetime.timedelta(seconds=dt / (epoch+1)))
                print(f"    epoch = {epoch}     dt={elapsedTime}    per-epoch={per_epoch}")
                # save fake.

                saveRawFile10(cur_step,
                              'fake_mr',
                              (fileStartVal + index * fileIncrement) / constVal,
                              '',
                              fake[0])

                saveRawFile10(cur_step, 'truth_mr', (fileStartVal + index * fileIncrement) / constVal, '',
                              mri[0])
                
                # show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                # show_tensor_images(real, size=(real_dim, target_shape, target_shape))
                # show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    saveModel(cur_step=cur_step)
            
            
                    
            cur_step += 1
    elapsed = time.time() - start_time
    elapsedTime = str(datetime.timedelta(seconds=elapsed))
    print(f"predict/infer time consumed: {elapsed} seconds, {elapsedTime}")
train()