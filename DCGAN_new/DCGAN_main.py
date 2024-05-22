import time
import tools
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
from DCGAN_model import ResUNet_LRes,Discriminator
from tqdm import tqdm
import os

# parameter for dataloader
data_path = "C:\\Files\\Research\\dataSet\\dataSet0"
data_save_path = "C:\\Files\\Research\\Volume_Inpainting\\DCGAN_new\\out"
volume_shape = (160,224,168)
target_shape = (128,128,128)
mask_type = "train"
data_type = np.float32

# parameter for network
gen_input_channel = 1
gen_dp_prob = 0.2
disc_input_channel = 2
learning_rate = 0.0002             #原模型参数 5e-3(0.005)
batch_size = 1
lambda_recon = 200

# parameter for train
save_model = True
save_raw = True

# tool functions
def save_raw_file(fileName, raw_file):
    # copy tensor from gpu to cpu.
    raw_file = raw_file.cpu()
    # convert tensor to numpy ndarray.
    raw_file = raw_file.detach().numpy()
    raw_file.astype('float32').tofile(fileName)


class DCGAN_Trainer:
    def __init__(self):
        self.model_name = "DCGAN"
        self.device = torch.device("cuda:0")
        self.total_epoch = 1000
        # 设置数据集
        self.dataset = tools.DataSet(data_path=data_path,
                                     volume_shape=volume_shape,
                                     target_shape=target_shape,
                                     mask_type=mask_type,
                                     data_type=np.float32)
        self.data_loader = DataLoader(dataset=self.dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)
        
        self.data_size = len(self.dataset)
        
        self.net_G = ResUNet_LRes(in_channel=gen_input_channel,dp_prob=gen_dp_prob).to(self.device)
        self.net_G_opt = torch.optim.Adam(self.net_G.parameters(), lr=learning_rate)
        self.net_D = Discriminator(disc_input_channel).to(self.device)
        self.net_D_opt = torch.optim.Adam(self.net_D.parameters(), lr=learning_rate)
        
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        
        
        
        
    def run(self):
        iter_counter=0
        for epoch_idx in tqdm(range(self.total_epoch),unit="epoch"):
            for ground_truth, mask in self.data_loader:
                # 初始化输入
                # ct=ground_truth*mask
                # mri=ground_truth
                masked_data = ground_truth*mask
                ground_truth=ground_truth.to(self.device)
                masked_data=masked_data.to(self.device)


                # 首先更新鉴别器
                with torch.no_grad():
                    # 在不记录梯度的情况下走一遍生成器
                    fake = self.net_G(masked_data)
                
                # 计算鉴别器损失
                D_fake_hat = self.net_D(fake.detach(),masked_data) # Detach generator
                D_fake_loss = self.net_D(D_fake_hat, torch.zeros_like(D_fake_hat))
                D_real_hat = self.net_D(ground_truth, masked_data)
                D_real_loss = self.net_D(D_real_hat, torch.ones_like(D_real_hat))
                D_loss = (D_fake_loss + D_real_loss) / 2
                
                # 对鉴别器反向传播
                self.net_D_opt.zero_grad() # Zero out the gradient before backpropagation
                D_loss.backward(retain_graph=True) # Update gradients
                self.net_D_opt.step() # Update optimizer

                # 更新生成器
                fake = self.net_G(masked_data)
                disc_fake_hat = self.net_D(fake, masked_data)
                gen_adv_loss = self.adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
                gen_rec_loss = self.recon_criterion(ground_truth, fake)
                gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
                
                # 对生成器反向传播
                self.net_G_opt.zero_grad()
                gen_loss.backward() # Update gradients
                self.net_G_opt.step() # Update optimizer

                # 保存和输出
                if save_model:
                    file_name = f"{self.model_name}_{epoch_idx}epoch_{iter_counter}iter.pth"
                    file_path = os.path.join(data_save_path,"weight",file_name)
                    torch.save({'gen': self.net_G.state_dict(),
                                'gen_opt': self.net_G_opt.state_dict(),
                                'disc': self.net_D.state_dict(),
                                'disc_opt': self.net_D_opt.state_dict(),
                                }, file_path)
                if save_raw:
                    save_object = ["ground_truth","masked_data","fake"]
                    variable_list = locals()
                    for item_name in save_object:
                        file_name = f"{variable_list[item_name]}_{epoch_idx}epoch_{iter_counter}iter.raw"
                        file_path = os.path.join(data_save_path,"output_data",file_name)
                        raw_file = variable_list[item_name][0].cpu()
                        raw_file = raw_file.detach().numpy()
                        raw_file.astype('float32').tofile(file_path)
                    
                    
                iter_counter += 1
                
                
                
    
        
        




if __name__ == '__main__':
    trainer = DCGAN_Trainer()
    trainer.run()