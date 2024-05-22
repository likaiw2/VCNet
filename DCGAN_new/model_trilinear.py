
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels):
        super(UNetUpResBlock, self).__init__()
        
        # trilinear + 11conv for upscale
        self.up_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)
        self.up_conv11 = nn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1,  stride=1, padding=0)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ELU()
        
        nn.init.xavier_uniform_(self.up_conv11.weight, gain = np.sqrt(2.0))
        nn.init.constant_(self.up_conv11.bias,0)
        
        self.resUnit = residualUnit(in_channels, out_channels, kernel_size=3)
        
    def forward(self,x,res):
        out=x+res
        out=self.up_conv11(self.up_tri_linear(out))
        out=self.activation(self.bn(out))
        
        out = torch.cat([out, res], 1)
        out = self.resUnit(out)

        return out


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


class ResUNet_LRes(nn.Module):
    def __init__(self, in_channel=1, out_channel=4, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
        # self.imsize = imsize

        self.activation = F.leaky_relu

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(3, stride=(2, 2, 2), padding=1)
        # self.pool4 = nn.MaxPool3d(2)

        # hidden_channel = 32
        hidden_channel = 16
        self.conv_block1_64 = UNetConvBlock(in_channel, hidden_channel)
        self.conv_block64_128 = residualUnit(hidden_channel, hidden_channel*2)
        self.conv_block128_256 = residualUnit(hidden_channel*2, hidden_channel*4)
        self.conv_block256_512 = residualUnit(hidden_channel*4, hidden_channel*8)
        # self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(hidden_channel*8, hidden_channel*4)
        self.up_block256_128 = UNetUpResBlock(hidden_channel*4, hidden_channel*2)
        self.up_block128_64 = UNetUpResBlock(hidden_channel*2, hidden_channel)
        self.Dropout = nn.Dropout3d(p=dp_prob)
        self.last = nn.Conv3d(hidden_channel, out_channel, 1, stride=1)

    # def forward(self, x, res_x):
    def forward(self, x):
        res_x = x
        block1 = self.conv_block1_64(x)             
        pool1 = self.pool1(block1)                  
        pool1_dp = self.Dropout(pool1)              
        
        block2 = self.conv_block64_128(pool1_dp)    
        pool2 = self.pool2(block2)                  
        pool2_dp = self.Dropout(pool2)              

        block3 = self.conv_block128_256(pool2_dp)   
        pool3 = self.pool3(block3)                 
        pool3_dp = self.Dropout(pool3)              

        block4 = self.conv_block256_512(pool3_dp)   
        
        # pool4 = self.pool4(block4)
        # pool4_dp = self.Dropout(pool4)
        # # block5 = self.conv_block512_1024(pool4_dp)
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        
        out = last
        
        out = torch.add(last, res_x)
        return out

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
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn



