import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import Volume_Inpainting.VCNet_modify.utils.tools as tools

class VoxelShuffle(nn.Module):
    def __init__(self,in_channels,out_channels,upscale_factor):
        super(VoxelShuffle,self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(in_channels,out_channels*(upscale_factor**3),kernel_size=3,stride=1,padding=1)
    
    def forward(self,x):
        x = self.voxel_shuffle(self.conv(x),self.upscale_factor)
        return x
    
    def voxel_shuffle(input, upscale_factor):
        batch_size, channels, in_height, in_width, in_depth = input.size()
        channels = channels // upscale_factor ** 3

        out_height = in_height * upscale_factor
        out_width = in_width * upscale_factor
        out_depth = in_depth * upscale_factor

        input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

        return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class Dilated_Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(Dilated_Block, self).__init__()
        
        # in the VCNet, in_channels and out_channels are the same
        self.conv1 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=2,padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=4,padding=4)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=8,padding=8)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.activation = nn.LeakyReLU()
        
        # nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv1.bias,0)
        # nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv2.bias,0)
        # nn.init.xavier_uniform_(self.conv3.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv3.bias,0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        # out += identity   #这个会报错，很呆，会产生inplace问题
        out = identity + out
        out = self.activation(out)

        return out

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        
        # this conv is for down sample
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=4, dilation=1, stride=2, padding=1)
        # this conv is for add channels
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=4, dilation=1,  stride=2, padding=1)
        self.activation1 = nn.ELU(inplace=True)
        self.activation2 = nn.ELU(inplace=True)

        # nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv1.bias,0)
        # nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv2.bias,0)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv1(2)
        out = self.activation2(out)
        out = self.bn(out)
        return out

class UpSampleBlock_T_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock_T_conv, self).__init__()
        
        # transposition for upscale
        self.up_t_conv = nn.ConvTranspose3d(in_channels, out_channels,kernel_size=4,stride=2,padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU()
        
        # nn.init.xavier_uniform_(self.up_t_conv.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.up_t_conv.bias,0)

        
    def forward(self,x):
        out=self.activation(self.bn(self.up_t_conv(x)))
        return out

class UpSampleBlock_VS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock_VS, self).__init__()
    
        # voxel shuffle
        self.up_VS = VoxelShuffle(in_channels, out_channels,  upscale_factor=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU()
            
    def forward(self, x):
        out=self.activation(self.bn(self.up_VS(x)))

        return out

class UpSampleBlock_Trilinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock_Trilinear, self).__init__()
        
        # trilinear + 11conv for upscale
        self.up_conv11 = nn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1,  stride=1, padding=0)
        self.up_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ELU()
        
        # nn.init.xavier_uniform_(self.up_conv11.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.up_conv11.bias,0)
        
    def forward(self,x,res):
        out=x+res
        out=self.up_tri_linear(out)
        out=self.up_conv11(out)
        out=self.activation(out)
        out=self.bn(out)

        return out

class gated_conv(torch.nn.Module):
    """
    Gated Convolution with spetral normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(gated_conv, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv3d = torch.nn.utils.spectral_norm(self.conv3d)
        self.mask_conv3d = torch.nn.utils.spectral_norm(self.mask_conv3d)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)
        #return torch.clamp(mask, -1, 1)

    def forward(self, input):
        x = self.conv3d(input)
        mask = self.mask_conv3d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm3d(x)
        else:
            return x




