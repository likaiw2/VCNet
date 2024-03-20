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
        
        nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.conv1.bias,0)
        nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.conv2.bias,0)
        nn.init.xavier_uniform_(self.conv3.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.conv3.bias,0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        # out = self.bn3(out)
        out = self.activation(out)

        # out += identity   #这个会报错，很呆，会产生inplace问题
        out = identity + out
        out = self.activation(out)

        return out

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        
        # this conv is for down sample
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=4, dilation=1, stride=2, padding=1)
        # this conv is for add channels
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=4, dilation=1,  stride=2, padding=1)
        self.activation = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.conv1.bias,0)
        nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.conv2.bias,0)
        
    def forward(self, x):
        # out = self.activation(self.bn1(self.conv1(x)))
        # out = self.activation(self.bn2(self.conv2(x)))
        # out = self.activation(self.pool(out))
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        return out

class UpSampleBlock_T_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock_T_conv, self).__init__()
        
        # transposition for upscale
        self.up_t_conv = nn.ConvTranspose3d(in_channels, out_channels,kernel_size=4,stride=2,padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU()
        
        nn.init.xavier_uniform_(self.up_t_conv.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.up_t_conv.bias,0)

        
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
        self.activation = nn.LeakyReLU()
        
        nn.init.xavier_uniform_(self.up_conv11.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.up_conv11.bias,0)
        
    def forward(self,x):
        # print("x:",x.shape)
        out=self.up_tri_linear(x)
        # print("up_tri_linear:",out.shape)
        out=self.up_conv11(out)
        # print("up_conv11:",out.shape)
        out=self.bn(out)
        # print("bn:",out.shape)
        out=self.activation(out)

        return out



class UNet_v2(nn.Module):
    def __init__(self, down_mode=3, up_mode=3):
        super(UNet_v2, self).__init__()
        # input = ["batch_size",1,128,128,128] 
        
        self.activate_fun = nn.Sigmoid()
        # Conv + ReLU (down sample)
        self.down_sample_1 = DownSampleBlock(in_channels=1,  out_channels=32)
        self.down_sample_2 = DownSampleBlock(in_channels=32, out_channels=64)
        self.down_sample_3 = DownSampleBlock(in_channels=64, out_channels=128)
        self.down_sample_4 = DownSampleBlock(in_channels=128,out_channels=256)
        
        # dilated conv + RB 作者表述不清不楚， 邮件询问后，得到三个 dilated RB 一模一样
        self.mid_1 = Dilated_Block(in_channels=256,out_channels=256)
        self.mid_2 = Dilated_Block(in_channels=256,out_channels=256)
        self.mid_3 = Dilated_Block(in_channels=256,out_channels=256)

        # upsample
        if up_mode == 1:
            # use transposition conv
            self.up_sample_4 = UpSampleBlock_T_conv(in_channels=256,out_channels=128)
            self.up_sample_3 = UpSampleBlock_T_conv(in_channels=128,out_channels=64)
            self.up_sample_2 = UpSampleBlock_T_conv(in_channels=64,out_channels=32)
            self.up_sample_1 = UpSampleBlock_T_conv(in_channels=32,out_channels=1)
        
        if up_mode == 2:
            # use Voxel Shuffle +31conv
            self.up_sample_4 = UpSampleBlock_VS(in_channels=256,out_channels=128)
            self.up_sample_3 = UpSampleBlock_VS(in_channels=128,out_channels=64)
            self.up_sample_2 = UpSampleBlock_VS(in_channels=64,out_channels=32)
            self.up_sample_1 = UpSampleBlock_VS(in_channels=32,out_channels=1)
        
        if up_mode == 3:
            # use Trilinear + 11conv
            self.up_sample_4 = UpSampleBlock_Trilinear(in_channels=256,out_channels=128)
            self.up_sample_3 = UpSampleBlock_Trilinear(in_channels=128,out_channels=64)
            self.up_sample_2 = UpSampleBlock_Trilinear(in_channels=64,out_channels=32)
            self.up_sample_1 = UpSampleBlock_Trilinear(in_channels=32,out_channels=1)

        self.up_res_conv_4 = nn.Conv3d(in_channels=256,  out_channels=128,  kernel_size=3, dilation=1,  stride=1, padding=1)
        self.up_res_conv_3 = nn.Conv3d(in_channels=128,  out_channels=64,  kernel_size=3, dilation=1,  stride=1, padding=1)
        self.up_res_conv_2 = nn.Conv3d(in_channels=64,  out_channels=32,  kernel_size=3, dilation=1,  stride=1, padding=1)
        self.up_res_conv_1 = nn.Conv3d(in_channels=1,  out_channels=1,  kernel_size=3, dilation=1,  stride=1, padding=1)
        nn.init.xavier_uniform_(self.up_res_conv_4.weight, gain = np.sqrt(2.0))
        nn.init.xavier_uniform_(self.up_res_conv_3.weight, gain = np.sqrt(2.0))
        nn.init.xavier_uniform_(self.up_res_conv_2.weight, gain = np.sqrt(2.0))
        nn.init.xavier_uniform_(self.up_res_conv_1.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.up_res_conv_4.bias,0)
        nn.init.constant_(self.up_res_conv_3.bias,0)
        nn.init.constant_(self.up_res_conv_2.bias,0)
        nn.init.constant_(self.up_res_conv_1.bias,0)
        
        self.up_bn1=nn.BatchNorm3d(128)
        self.up_bn2=nn.BatchNorm3d(64)
        self.up_bn3=nn.BatchNorm3d(32)
        self.up_bn4=nn.BatchNorm3d(1)

        self.final_activate_fun = nn.Tanh()
        
    def forward(self, x, test_mode=False, dataSavePath="/home/dell/storage/WANGLIKAI/VCNet/output"):
        res_x = x
        
        # down_sample_1     1,128,128->32,64,64
        out = self.down_sample_1(x)
        # print("down_sample_1:",out.shape)
        res_1 = out
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_64",f"down_64_{i}",out[0, i, :, :, :])
        
        # down_sample_2     32,64,64->64,32,32
        out = self.down_sample_2(out)
        # print("down_sample_2:",out.shape)
        res_2 = out
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_32",f"down_32{i}",out[0, i, :, :, :])
        
        # down_sample_3     64,32,32->128,16,16
        out = self.down_sample_3(out)
        # print("down_sample_3:",out.shape)
        res_3 = out
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_16",f"down_16{i}",out[0, i, :, :, :])

        # down_sample_4     128,16,16->256,8,8
        out = self.down_sample_4(out)
        # print("down_sample_4:",out.shape)
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#down_8",f"testRAW_{i}",out[0, i, :, :, :])
        
        # mid conv + RB 作者表述不清不楚，目前暂定三个 dilated RB 一模一样
        out=self.mid_1(out)
        # print("mid_1",out.shape)
        out=self.mid_2(out)
        # print("mid_2",out.shape)
        out=self.mid_3(out)
        # print("mid_3", out.shape, "\n")
        
        # up_sample_4       256,8,8->128,16,16
        out=self.up_sample_4(out)
        # print("up_sample_4:",out.shape)
        out=torch.cat([out, res_3], dim=1)
        # out=self.activate_fun(self.bn4(self.up_res_conv_4(out)))
        out=self.activate_fun(self.up_res_conv_4(out))
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#up_16",f"up_16_{i}",out[0, i, :, :, :])
        # print("layer4_conv",out.shape)
        
        # up_sample_3       128,16,16->64,32,32
        out=self.up_sample_3(out)
        out=torch.cat([out, res_2], dim=1)
        # print("layer3_cat",out.shape)
        # out=self.activate_fun(self.bn3(self.up_res_conv_3(out)))
        out=self.activate_fun(self.up_res_conv_3(out))
        # print("layer3_conv",out.shape)
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#up_32",f"up_32_{i}",out[0, i, :, :, :])
        
        # up_sample_2       64,32,32->32,64,64
        out=self.up_sample_2(out)
        out=torch.cat([out, res_1], dim=1)
        # print("layer2_cat",out.shape)
        # out=self.activate_fun(self.bn2(self.up_res_conv_2(out)))
        out=self.activate_fun(self.up_res_conv_2(out))
        # print("layer2_conv",out.shape)
        if test_mode:
            for i in range(32):
                tools.saveRawFile10(f"{dataSavePath}/#up_64",f"up_64_{i}",out[0, i, :, :, :])
        
        # up_sample_1       32,64,64->1,128,128
        out=self.up_sample_1(out)
        # print("up_sample_1:",out.shape)
        # out=torch.cat([out, res_x], dim=1)
        # out=self.final_activate_fun(self.bn1(self.up_res_conv_1(out)))
        out=self.final_activate_fun(self.up_res_conv_1(out))
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
    
