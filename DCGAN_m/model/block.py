import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


#------------------原论文中的block（始）------------------
# Ordinary UNet Conv Block 卷积块
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.leaky_relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation


        nn.init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        nn.init.constant(self.conv.bias,0)
        nn.init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0))
        nn.init.constant(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out

# two-layer residual unit: two conv with BN/leaky_relu and identity mapping 残差单元
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.leaky_relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        nn.init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        nn.init.constant(self.conv2.bias, 0)
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
        nn.init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        nn.init.constant(self.up.bias, 0)

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

class UNetUpResBlock_223(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.leaky_relu, space_dropout=False):
        super(UNetUpResBlock_223, self).__init__()
        # 转置卷积 输入
        # print("before size:",in_size,out_size)
        self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=3, stride=(2, 2, 3), padding=1, output_padding=(1,1,2))
        # (original) kernel_size=2 out_padding=0

        # print("after size:",in_size,out_size)

        self.bnup = nn.BatchNorm3d(out_size)
        nn.init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        nn.init.constant(self.up.bias, 0)

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
#------------------原论文中的block（末）------------------

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def     __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        #self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        if use_bn:
            #self.batchnorm = nn.BatchNorm2d(input_channels * 2)
            self.batchnorm = nn.BatchNorm3d(input_channels * 2)
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
