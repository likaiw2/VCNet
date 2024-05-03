import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils.tools as tools
from model.blocks import * 
from torchvision import models


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
    

class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=1, upsampling_mode='trilinear'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsample_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        self.enc_5 = PCBActiv(512, 512, sample='down-3')
        self.enc_6 = PCBActiv(512, 512, sample='down-3')
        self.enc_7 = PCBActiv(512, 512, sample='down-3')

        self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        
        # 初始输入和遮罩
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        
        # 编码器
        h_dict['h_1'], h_mask_dict['h_1'] = self.enc_1(h_dict['h_0'], h_mask_dict['h_0'])
        h_dict['h_2'], h_mask_dict['h_2'] = self.enc_2(h_dict['h_1'], h_mask_dict['h_1'])
        h_dict['h_3'], h_mask_dict['h_3'] = self.enc_3(h_dict['h_2'], h_mask_dict['h_2'])
        h_dict['h_4'], h_mask_dict['h_4'] = self.enc_4(h_dict['h_3'], h_mask_dict['h_3'])
        h_dict['h_5'], h_mask_dict['h_5'] = self.enc_5(h_dict['h_4'], h_mask_dict['h_4'])
        h_dict['h_6'], h_mask_dict['h_6'] = self.enc_6(h_dict['h_5'], h_mask_dict['h_5'])
        # h_dict['h_7'], h_mask_dict['h_7'] = self.enc_7(h_dict['h_6'], h_mask_dict['h_6'])
        
        # 保存一下，第七层就是最底层
        # h, h_mask = h_dict['h_7'], h_mask_dict['h_7']
        h, h_mask = h_dict['h_6'], h_mask_dict['h_6']

        # 解码器
        # h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_6')
        # h,h_mask = self.dec_7(h, h_mask)
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_5')
        h, h_mask = self.dec_6(h, h_mask)
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_4')
        h, h_mask = self.dec_5(h, h_mask)
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_3')
        h, h_mask = self.dec_4(h, h_mask)
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_2')
        h, h_mask = self.dec_3(h, h_mask)
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_1')
        h, h_mask = self.dec_2(h, h_mask)

        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_0')
        h, h_mask = self.dec_1(h, h_mask)

        return h+input, h_mask
    
    def up_sample(self,h,h_mask,h_dict,h_mask_dict,layer_name):
        
        h = F.interpolate(h, scale_factor=2, mode=self.upsample_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')
        # h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsample_mode)
        
        h = torch.cat([h, h_dict[layer_name]], dim=1)
        h_mask = torch.cat([h_mask, h_mask_dict[layer_name]], dim=1)
        
        return h,h_mask


class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, n_in_channel=5):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            #downsample to 64
            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            #Self_Attn(2*cnum, 'relu'),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            #Self_Attn(cnum//2, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample
            GatedConv2dWithActivation(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )
        self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=False)
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            #Self_Attn(cnum, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )


    def forward(self, imgs, masks, img_exs=None):
        # Coarse
        masked_imgs =  imgs * (1 - masks) + masks
        if img_exs == None:
            input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        #print(input_imgs.size(), imgs.size(), masks.size())
        x = self.coarse_net(input_imgs)
        x = torch.clamp(x, -1., 1.)
        coarse_x = x
        # Refine
        masked_imgs = imgs * (1 - masks) + coarse_x * masks
        if img_exs is None:
            input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        x = self.refine_conv_net(input_imgs)
        x= self.refine_attn(x)
        #print(x.size(), attention.size())
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)
        return coarse_x, x

class InpaintSADirciminator(nn.Module):
    def __init__(self):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(5, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            Self_Attn(8*cnum, 'relu'),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x



