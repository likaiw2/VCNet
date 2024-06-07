import datetime
import time
import models.model_VCNet
import models.model_gated
import models.model_p2p
import models.model
import models.model_deep
import models.model_deep_partial
import tools
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
import os
import wandb
import evaluation

from config import get_cfg_defaults
cfg = get_cfg_defaults()

# parameter
test_data_path="/root/autodl-fs/brain_168_224_160_mr/test"
pth_path="/root/autodl-tmp/Diode/Codes/Volume_Impainting/DCGAN_new/out/VCNet_06080030/weight/VCNet_06080030_0epoch_1iter.pth"
data_save_path="/root/autodl-tmp/Diode/Codes/Volume_Impainting/DCGAN_new/out"
# volume_shape=(128,128,128)
volume_shape=(160,224,168)
target_shape=(128,128,128)
device = torch.device("cuda:0")
model_name="MyNet"

# make data loader
test_dataset = tools.DataSet(data_path=cfg.dataset.test_data_path,
                             volume_shape=volume_shape,
                             target_shape=target_shape,
                             data_type=np.float32)
test_data_loader = DataLoader(dataset=test_dataset,batch_size=1,
                              shuffle=True,
                              num_workers=1)
test_data_size = len(test_data_loader)

gt,mask = test_dataset[0]
mask = torch.Tensor(mask.reshape((1,1,128,128,128)))
mask = mask.to(device)

# init gen and disc
# Generator = models.model.ResUNet_LRes(1,1,0.2)
# Discriminator = models.model.Discriminator(2)

Generator = models.model_VCNet.UNet_v2(down_mode=3,up_mode=1)
Discriminator = models.model.Discriminator(2)


net_G = Generator.to(device)

if os.path.exists(pth_path):
    loaded_state = torch.load(pth_path)
    # print(loaded_state["gen"].keys())
    net_G.load_state_dict(loaded_state["gen"])
    print("Weight load success!")
else:
    print("load weights failed!")
    
iter=0
for ground_truth, _ in test_data_loader:
    # 初始化输入
    ground_truth=ground_truth.to(device)
    masked_data = ground_truth*mask
    
    masked_data=masked_data.to(device)
    
    truth = ground_truth
    masked = masked_data
    
    # 首先更新鉴别器
    with torch.no_grad():
        # 在不记录梯度的情况下走一遍生成器
        fake = net_G(masked_data,mask)
        

    # 保存和输出
    save_object = ["truth","masked","fake"]
    variable_list = locals()
    
    for item_name in save_object:
        file_name = f"{model_name}_{iter}_{item_name}.raw"
        folder_path = os.path.join(data_save_path,model_name,"test_out")
        os.makedirs(folder_path) if not os.path.isdir(folder_path) else None
        file_path = os.path.join(folder_path,file_name)
        print(file_path)
        raw_file = variable_list[item_name][0].cpu()
        raw_file = raw_file.detach().numpy()
        raw_file.astype('float32').tofile(file_path)
        
    iter+=1
