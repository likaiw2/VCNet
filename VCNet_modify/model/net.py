import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import Volume_Inpainting.VCNet_modify.utils.tools as tools
from Volume_Inpainting.VCNet_modify.model.blocks import * 

class UNet_v2(nn.Module):
    def __init__(self, down_mode=3, up_mode=3):
        super(UNet_v2, self).__init__()
        # input = ["batch_size",1,128,128,128] 
        
        self.activate_fun = nn.LeakyReLU()
        # Conv + ReLU (down sample)
        self.down_sample_1 = DownSampleBlock(in_channels=1,  out_channels=32)
        self.down_sample_2 = DownSampleBlock(in_channels=32, out_channels=64)
        self.down_sample_3 = DownSampleBlock(in_channels=64, out_channels=128)
        
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

        self.final_activate_fun = nn.tanh()
        # self.final_activate_fun = tools.Swish(0.5)
        
    def forward(self, x,mask, test_mode=False, dataSavePath="/home/dell/storage/WANGLIKAI/VCNet/output"):
        res_0 = x
        x = torch.cat([x, mask], dim=1)
        
        # down_sample_1     2,128,128->32,64,64
        out = self.down_sample_1(x)
        # print("down_sample_1:",out.shape)
        res_1 = out
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#down_64",f"down_64_{i}",out[0, i, :, :, :])
        
        # down_sample_2     32,64,64->64,32,32
        out = self.down_sample_2(out)
        # print("down_sample_2:",out.shape)
        res_2 = out
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#down_32",f"down_32{i}",out[0, i, :, :, :])
        
        # down_sample_3     64,32,32->128,16,16
        out = self.down_sample_3(out)
        # print("down_sample_3:",out.shape)
        res_3 = out
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#down_16",f"down_16{i}",out[0, i, :, :, :])

        # mid conv + RB 作者表述不清不楚，目前暂定三个 dilated RB 一模一样
        out=self.mid_1(out)
        out=self.mid_2(out)
        out=self.mid_3(out)
        
        # up_sample_3       128,16,16->64,32,32
        out=self.up_sample_3(out,res_3)
        # print("layer3_conv",out.shape)
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#up_32",f"up_32_{i}",out[0, i, :, :, :])
        
        # up_sample_2       64,32,32->32,64,64
        out=self.up_sample_2(out,res_2)
        # print("layer2_conv",out.shape)
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#up_64",f"up_64_{i}",out[0, i, :, :, :])
        
        # up_sample_1       32,64,64->1,128,128
        out=self.up_sample_1(out,res_1)
        # print("up_sample_1:",out.shape)
        
        
        out=self.final_activate_fun(self.up_bn1(self.up_res_conv_1(out)))
        # out=self.final_activate_fun(self.up_bn1(self.up_res_conv_1(out)))
        # print("layer1_conv(final)",out.shape)
        
        return out
    
class Dis_VCNet(nn.Module):
    def __init__(self):
        super(Dis_VCNet,self).__init__()

        self.activate_fun = nn.ReLU(inplace=True)   # 原地修改数据，可以节省空间

        self.start_conv = nn.Conv3d(in_channels=1,   out_channels=32, kernel_size=1)
        self.down_1_conv = nn.Conv3d(in_channels=1,   out_channels=32,  kernel_size=4, dilation=1,  stride=2)
        self.down_2_conv = nn.Conv3d(in_channels=32,  out_channels=64,  kernel_size=4, dilation=1,  stride=2)
        self.down_3_conv = nn.Conv3d(in_channels=64,  out_channels=128,  kernel_size=4, dilation=1,  stride=2)
        self.down_4_conv = nn.Conv3d(in_channels=128,   out_channels=1,  kernel_size=4, dilation=1,  stride=2)
        self.avg = nn.AdaptiveAvgPool3d(output_size=1)
        
        self.pool1 = nn.MaxPool3d(kernel_size=4,stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=4,stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=4,stride=2)
        self.pool4 = nn.MaxPool3d(kernel_size=4,stride=2)
        
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(1)
        self.activate_fun = nn.LeakyReLU()

    def forward(self,x):
        # out = self.activate_fun(self.start_conv(x))
        # print("x:",x.shape)
        out = self.activate_fun(self.down_1_conv(x))
        # out = self.pool1(out)
        out = self.activate_fun(self.down_2_conv(out))
        # out = self.pool2(out)
        out = self.activate_fun(self.down_3_conv(out))
        # out = self.pool3(out)
        out = self.activate_fun(self.down_4_conv(out))
        # out = self.pool4(out)
        out = self.avg(out)
        out = self.activate_fun(out)
        
        return out
    
