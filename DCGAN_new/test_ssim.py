import pytorch_ssim
import torch
from torch.autograd import Variable
import numpy as np

img1 = Variable(torch.rand(1, 1, 256, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256, 256))

gt = "/Users/wanglikai/Codes/test/data/ground_truth_0523_999epoch_69999iter.raw"
fk = "/Users/wanglikai/Codes/test/data/fake_0523_999epoch_69999iter.raw"
gt = np.fromfile(gt, dtype=np.float32).reshape((1,1,128,128,128))
fk = np.fromfile(fk, dtype=np.float32).reshape((1,1,128,128,128))

gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt))*255.0
fk = (fk-np.min(fk))/(np.max(fk)-np.min(fk))*255.0

gt = torch.from_numpy(gt)
fk = torch.from_numpy(fk)

print(pytorch_ssim.ssim3D(gt, fk))