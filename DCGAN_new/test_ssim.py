import pytorch_ssim
import torch
from torch.autograd import Variable
import numpy as np

img1 = Variable(torch.rand(1, 1, 256, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256, 256))

gt = "/root/autodl-tmp/Diode/Codes/Volume_Impainting/DCGAN_new/out/DCGAN_origional/output/999epoch_69999iter_truth_0524.raw"
fk = "/root/autodl-tmp/Diode/Codes/Volume_Impainting/DCGAN_new/out/DCGAN_origional/output/999epoch_69999iter_fake_0524.raw"

gt = np.fromfile(gt, dtype=np.float32).reshape((1,1,128,128,128))
fk = np.fromfile(fk, dtype=np.float32).reshape((1,1,128,128,128))

gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt))*255.0
fk = (fk-np.min(fk))/(np.max(fk)-np.min(fk))*255.0

gt = torch.from_numpy(gt)
fk = torch.from_numpy(fk)

print(pytorch_ssim.ssim3D(gt, fk))