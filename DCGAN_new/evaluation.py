import numpy as np
from skimage.metrics import structural_similarity as ssim

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

def get_ssim(im1,im2):
    data_range=255.0
    # 归一化到指定大小 0 - data_range
    img1 = ((im1-np.min(im1))/(np.max(im1)-np.min(im1)))*data_range
    img2 = ((im2-np.min(im2))/(np.max(im2)-np.min(im2)))*data_range
    
    ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min())
    
    return ssim_value