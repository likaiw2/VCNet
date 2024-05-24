import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




def get_mse(im1,im2):
    data_range=255.0
    # 归一化到指定大小 0 - data_range
    data1 = ((im1-np.min(im1))/(np.max(im1)-np.min(im1)))*data_range
    data2 = ((im2-np.min(im2))/(np.max(im2)-np.min(im2)))*data_range
    
    mse = ((data1 - data2) ** 2).mean()
    return mse

def get_psnr(im1, im2):
    data_range=255.0
    # 归一化到指定大小 0 - data_range
    data1 = ((im1-np.min(im1))/(np.max(im1)-np.min(im1)))*data_range
    data2 = ((im2-np.min(im2))/(np.max(im2)-np.min(im2)))*data_range
    
    mse = ((data1 - data2) ** 2).mean()
    psnr = 10 * np.log10(data_range * data_range / mse)
    
    return psnr

def get_ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)