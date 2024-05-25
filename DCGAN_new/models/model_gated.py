import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import tools

# Ordinary UNet Conv Block 卷积块
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.leaky_relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation


        nn.init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.conv.bias,0)
        nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out

class Tri_UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tri_UpResBlock, self).__init__()
        
        # trilinear + 11conv for upscale
        self.up_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)
        self.up_conv11 = nn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1,  stride=1, padding=0)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ELU()
        
        nn.init.xavier_uniform_(self.up_conv11.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.up_conv11.bias,0)
        
        self.resUnit = residualUnit(in_channels, out_channels, kernel_size=3)
        
    def forward(self,x,res):
        out=self.up_conv11(self.up_tri_linear(x))
        out=self.activation(self.bn(out))
        
        out = torch.cat([out, res], 1)
        out = self.resUnit(out)

        return out

# two-layer residual unit: two conv with BN/leaky_relu and identity mapping 残差单元
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.leaky_relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        nn.init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        nn.init.constant_(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm3d(out_size)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm3d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn2(self.conv2(out1)))
        if self.in_size != self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output

# Ordinary Residual UNet-Up Conv Block
class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.leaky_relu, space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        # 转置卷积 输入
        # print("before size:",in_size,out_size)
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)  # 为了抑制棋盘效应，改成了(1,1)发现会报错  #7.11改成3,3（依旧报错）
        # print("after size:",in_size,out_size)

        self.bnup = nn.BatchNorm3d(out_size)
        nn.init.xavier_uniform_(self.up.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.up.bias, 0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size=kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        # print ('    x.shape: ',x.shape, ' \n    bridge.shape: ',bridge.shape)
        up = self.activation(self.bnup(self.up(x)))

        # crop1 = self.center_crop(bridge, up.size()[2])
        # print ('    up.shape: ',up.shape, ' \n    crop1.shape: ',crop1.shape)

        crop1 = bridge
        # print ('    up.shape: ',up.shape, ' \n    crop1.shape: ',crop1.shape)

        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        #self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        if use_bn:
            #self.batchnorm = nn.Batchnorm3d(input_channels * 2)
            self.batchnorm = nn.BatchNorm3d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
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
        # self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
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

class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(DilatedBlock, self).__init__()
        
        # in the VCNet, in_channels and out_channels are the same
        self.conv1 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=2,padding=tools.get_pad(16, 3, 1, 2))
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=4,padding=tools.get_pad(16, 3, 1, 4))
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=8,padding=tools.get_pad(16, 3, 1, 8))
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

class GatedConv3dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv3d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv3dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm3d = torch.nn.Batchnorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
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

class GatedDeConv3dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv3d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv3dWithActivation, self).__init__()
        self.conv3d = GatedConv3dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv3d(x)

class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, n_in_channel=5):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv3dWithActivation(n_in_channel, cnum, 5, 1, padding=tools.get_pad(256, 5, 1)),
            # downsample 128
            GatedConv3dWithActivation(cnum, 2*cnum, 4, 2, padding=tools.get_pad(256, 4, 2)),
            GatedConv3dWithActivation(2*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            #downsample to 64
            GatedConv3dWithActivation(2*cnum, 4*cnum, 4, 2, padding=tools.get_pad(128, 4, 2)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=tools.get_pad(64, 3, 1, 2)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            # upsample
            GatedDeConv3dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            #Self_Attn(2*cnum, 'relu'),
            GatedConv3dWithActivation(2*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            GatedDeConv3dWithActivation(2, 2*cnum, cnum, 3, 1, padding=tools.get_pad(256, 3, 1)),

            GatedConv3dWithActivation(cnum, cnum//2, 3, 1, padding=tools.get_pad(256, 3, 1)),
            #Self_Attn(cnum//2, 'relu'),
            GatedConv3dWithActivation(cnum//2, 3, 3, 1, padding=tools.get_pad(128, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv3dWithActivation(n_in_channel, cnum, 5, 1, padding=tools.get_pad(256, 5, 1)),
            # downsample
            GatedConv3dWithActivation(cnum, cnum, 4, 2, padding=tools.get_pad(256, 4, 2)),
            GatedConv3dWithActivation(cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            # downsample
            GatedConv3dWithActivation(2*cnum, 2*cnum, 4, 2, padding=tools.get_pad(128, 4, 2)),
            GatedConv3dWithActivation(2*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=tools.get_pad(64, 3, 1, 2)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=tools.get_pad(64, 3, 1, 4)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=tools.get_pad(64, 3, 1, 8)),

            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=tools.get_pad(64, 3, 1, 16))
        )
        # self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=False)
        self.refine_upsample_net = nn.Sequential(
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),

            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedDeConv3dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            GatedConv3dWithActivation(2*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            GatedDeConv3dWithActivation(2, 2*cnum, cnum, 3, 1, padding=tools.get_pad(256, 3, 1)),

            GatedConv3dWithActivation(cnum, cnum//2, 3, 1, padding=tools.get_pad(256, 3, 1)),
            #Self_Attn(cnum, 'relu'),
            GatedConv3dWithActivation(cnum//2, 3, 3, 1, padding=tools.get_pad(256, 3, 1), activation=None),
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
    
class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.final = nn.Conv3d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn



