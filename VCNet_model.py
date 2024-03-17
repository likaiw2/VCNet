import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import tools



def voxel_shuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels = channels // upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class VoxelShuffle(nn.Module):
	def __init__(self,in_channels,out_channels,upscale_factor):
		super(VoxelShuffle,self).__init__()
		self.upscale_factor = upscale_factor
		self.conv = nn.Conv3d(in_channels,out_channels*(upscale_factor**3),kernel_size=3,stride=1,padding=1)

	def forward(self,x):
		x = voxel_shuffle(self.conv(x),self.upscale_factor)
		return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        # in the VCNet, in_channels and out_channels are the same
        self.conv1 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=2,padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=4,padding=4)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=8,padding=8)
        self.bn3 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # out += identity   #这个会报错，很呆，会产生inplace问题
        out = identity + out
        out = self.relu(out)

        return out

class UNet_v2(nn.Module):
    def __init__(self, in_channel=1):   #n_classes 不知道干啥用的我给删掉了
        super(UNet_v2, self).__init__()
        input_feathure = [2,1,128,128,128]
                             
        self.activate_fun = nn.ReLU(inplace=True)   # 原地修改数据，可以节省空间
        
        # Conv + ReLU (down sample)
        self.down_1_conv1 = nn.Conv3d(in_channels=32,   out_channels=32,  kernel_size=4, dilation=1,  stride=2, padding=1)
        self.down_1_conv2 = nn.Conv3d(in_channels=1,  out_channels=32,  kernel_size=3, dilation=1,  stride=1, padding=1)
        
        self.down_2_conv1 = nn.Conv3d(in_channels=64,  out_channels=64,  kernel_size=4, dilation=1,  stride=2, padding=1)
        self.down_2_conv2 = nn.Conv3d(in_channels=32,  out_channels=64,  kernel_size=3, dilation=1,  stride=1, padding=1)
        
        self.down_3_conv1 = nn.Conv3d(in_channels=128,  out_channels=128, kernel_size=4, dilation=1,  stride=2, padding=1)
        self.down_3_conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, dilation=1,  stride=1, padding=1)
        
        self.down_4_conv1 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=4, dilation=1,  stride=2, padding=1)
        self.down_4_conv2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, dilation=1,  stride=1, padding=1)

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.pool4 = nn.MaxPool3d(2)
        
        # dilated conv + RB 作者表述不清不楚， 邮件询问后，得到三个 dilated RB 一模一样
        self.mid_middle1 = ResidualBlock(in_channels=256,out_channels=256)
        self.mid_middle2 = ResidualBlock(in_channels=256,out_channels=256)
        self.mid_middle3 = ResidualBlock(in_channels=256,out_channels=256)

        # VS+Conv+ReLU
        self.up_4_tconv = nn.ConvTranspose3d(in_channels=256, out_channels=128,kernel_size=4,stride=2,padding=1)
        self.up_4_VS = VoxelShuffle(in_channels=256, out_channels=128,  upscale_factor=2)
        self.up_4_conv = nn.Conv3d(in_channels=256,  out_channels=128,  kernel_size=3, dilation=1,  stride=1, padding=1)
        self.up_4_conv11 = nn.Conv3d(in_channels=256,  out_channels=128,  kernel_size=1, dilation=1,  stride=1, padding=0)
        self.up_4_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)


        self.up_3_tconv = nn.ConvTranspose3d(in_channels=128, out_channels=64,kernel_size=4,stride=2,padding=1)
        self.up_3_VS = VoxelShuffle(in_channels=128, out_channels=64,   upscale_factor=2)
        self.up_3_conv = nn.Conv3d(in_channels=128,  out_channels=64,   kernel_size=3, dilation=1,  stride=1, padding=1)
        self.up_3_conv11 = nn.Conv3d(in_channels=128,  out_channels=64,  kernel_size=1, dilation=1,  stride=1, padding=0)
        self.up_3_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)

        
        self.up_2_tconv = nn.ConvTranspose3d(in_channels=64, out_channels=32,kernel_size=4,stride=2,padding=1)
        self.up_2_VS = VoxelShuffle(in_channels=64,  out_channels=32,   upscale_factor=2)
        self.up_2_conv = nn.Conv3d(in_channels=64,   out_channels=32,   kernel_size=3, dilation=1,  stride=1, padding=1)
        self.up_2_conv11 = nn.Conv3d(in_channels=64,  out_channels=32,  kernel_size=1, dilation=1,  stride=1, padding=0)
        self.up_2_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)

        
        self.up_1_tconv = nn.ConvTranspose3d(in_channels=32, out_channels=1,kernel_size=4,stride=2,padding=1)
        self.up_1_VS = VoxelShuffle(in_channels=32,  out_channels=1,    upscale_factor=2)
        self.up_1_conv = nn.Conv3d(in_channels=1,    out_channels=1,    kernel_size=3, dilation=1,  stride=1, padding=1)
        self.up_1_conv11 = nn.Conv3d(in_channels=32,  out_channels=1,  kernel_size=1, dilation=1,  stride=1, padding=0)
        self.up_1_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)

        self.final_activate_fun = nn.Tanh()
        
    def forward(self, x, test_mode=False, VS_upscale=True, dataSavePath="/home/dell/storage/WANGLIKAI/VCNet/output"):
        res_x = x

        # Conv + ReLU (down sample)
        out=self.activate_fun(self.down_1_conv2(x))
        # print("layer1_conv1",out.shape)
        out=self.activate_fun(self.pool1(out))
        # print("layer1_conv2",out.shape)
        res_1 = out
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_64",f"testRAW_{i}",out[0, i, :, :, :])

        
        out=self.activate_fun(self.down_2_conv2(out))
        # print("layer2_conv1",out.shape)
        out=self.activate_fun(self.pool2(out))
        # print("layer2_conv2",out.shape)
        res_2 = out
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_32",f"testRAW_{i}",out[0, i, :, :, :])

        
        out=self.activate_fun(self.down_3_conv2(out))
        # print("layer3_conv1",out.shape)
        out=self.activate_fun(self.pool3(out))
        # print("layer3_conv2",out.shape)
        res_3 = out
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_16",f"testRAW_{i}",out[0, i, :, :, :])

        
        out=self.activate_fun(self.down_4_conv2(out))
        # print("layer4_conv1",out.shape)
        out=self.activate_fun(self.pool4(out))
        # print("layer4_conv2",out.shape,"\n")
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_8",f"testRAW_{i}",out[0, i, :, :, :])
        
        # dilated conv + RB 作者表述不清不楚，目前暂定三个 dilated RB 一模一样
        out=self.mid_middle1(out)
        # print("mid_1",out.shape)
        out=self.mid_middle2(out)
        # print("mid_2",out.shape)
        out=self.mid_middle3(out)
        # print("mid_3", out.shape, "\n")
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#mid",f"testRAW_{i}",out[0, i, :, :, :])

        
        # VS+Conv+ReLU
        if VS_upscale:
            out=self.activate_fun(self.up_4_VS(out))
        else:
            out=self.activate_fun(self.up_4_conv11(self.up_4_tri_linear(out)))
        # print("layer4_VS",out.shape)
        out=torch.cat([out, res_3], dim=1)
        # print("layer4_cat",out.shape)
        out=self.activate_fun(self.up_4_conv(out))
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#up_16",f"testRAW_{i}",out[0, i, :, :, :])
        # print("layer4_conv",out.shape)
        
        if VS_upscale:
            out=self.activate_fun(self.up_3_VS(out))
        else:
            # out=self.activate_fun(self.up_3_tconv(out))
            out=self.activate_fun(self.up_3_conv11(self.up_3_tri_linear(out)))

        # print("layer3_VS",out.shape)
        out=torch.cat([out, res_2], dim=1)
        # print("layer3_cat",out.shape)
        out=self.activate_fun(self.up_3_conv(out))
        # print("layer3_conv",out.shape)
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#up_32",f"testRAW_{i}",out[0, i, :, :, :])
        
        
        if VS_upscale:
            out=self.activate_fun(self.up_2_VS(out))
        else:
            # out=self.activate_fun(self.up_2_tconv(out))
            out=self.activate_fun(self.up_2_conv11(self.up_2_tri_linear(out)))
        # print("layer2_VS",out.shape)
        out=torch.cat([out, res_1], dim=1)
        # print("layer2_cat",out.shape)
        out=self.activate_fun(self.up_2_conv(out))
        # print("layer2_conv",out.shape)
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#up_64",f"testRAW_{i}",out[0, i, :, :, :])
        
        
        if VS_upscale:
            out=self.activate_fun(self.up_1_VS(out))
        else:
            # out=self.activate_fun(self.up_1_tconv(out))
            out=self.activate_fun(self.up_1_conv11(self.up_1_tri_linear(out)))
        # print("layer1_VS",out.shape)
        out=self.final_activate_fun(self.up_1_conv(out))
        # print("layer1_conv(final)",out.shape)
        
        return out
    
    
    
class Dis_VCNet(nn.Module):
    def __init__(self):
        super(Dis_VCNet,self).__init__()

        self.activate_fun = nn.ReLU(inplace=True)   # 原地修改数据，可以节省空间

        self.down_1_conv = nn.Conv3d(in_channels=1,   out_channels=32,  kernel_size=4, dilation=1,  stride=2)
        self.down_2_conv = nn.Conv3d(in_channels=32,  out_channels=64,  kernel_size=4, dilation=1,  stride=2)
        self.down_3_conv = nn.Conv3d(in_channels=64,  out_channels=128,  kernel_size=4, dilation=1,  stride=2)
        self.down_4_conv = nn.Conv3d(in_channels=128,   out_channels=1,  kernel_size=4, dilation=1,  stride=2)
        self.avg = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self,x):
        out = self.activate_fun(self.down_1_conv(x))
        out = self.activate_fun(self.down_2_conv(out))
        out = self.activate_fun(self.down_3_conv(out))
        out = self.activate_fun(self.down_4_conv(out))
        
        out = self.avg(out)
        
        return out
    
