import datetime
import time
import tools
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
from model_trilinear import ResUNet_LRes,Discriminator
from tqdm import tqdm
import os
import wandb

from config import get_cfg_defaults  # 导入获取默认配置的函数
cfg = get_cfg_defaults()

# parameter for network
gen_input_channel = cfg.net.gen_input_channel
gen_dp_prob = cfg.net.gen_dp_prob
disc_input_channel = cfg.net.disc_input_channel
learning_rate = cfg.net.learning_rate             #原模型参数 5e-3(0.005)
batch_size = cfg.net.batch_size
lambda_recon = cfg.net.lambda_recon

data_save_path = cfg.dataset.data_save_path

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
    def __init__(self,cfg):
        self.cfg = cfg
        self.model_name = self.cfg.net.model_name
        self.device = torch.device(self.cfg.train.device)
        self.total_epoch = self.cfg.train.total_epoch
        self.volume_shape = self.cfg.dataset.volume_shape
        self.target_shape = self.cfg.dataset.target_shape
        self.mask_type = self.cfg.dataset.mask_type
        
        if self.cfg.WANDB.WORK:
            self.cfg.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
            cfg.freeze()
            self.wandb = wandb
            self.wandb.init(project="volume_inpainting",
                            name=f"{self.cfe.net.model_name}_{datetime.datetime.now().strftime('%m%d_%H_%M')}",
                            notes=self.cfg.WANDB.LOG_DIR,
                            config=self.cfg,
                            mode="offline")
        
        # 设置数据集
        self.dataset = tools.DataSet(data_path=self.cfg.dataset.train_data_path,
                                     volume_shape=self.volume_shape,
                                     target_shape=self.target_shape,
                                     mask_type=self.mask_type,
                                     data_type=np.float32)
        self.data_loader = DataLoader(dataset=self.dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)
        self.data_size = len(self.dataset)
        self.display_step = np.ceil(np.ceil(self.data_size / batch_size) * self.total_epoch / 20)   #一共输出20个epoch，供判断用
        
        # 生成器鉴别器初始化
        def weights_init(m):
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)
        self.net_G = ResUNet_LRes(in_channel=gen_input_channel,out_channel=1,dp_prob=gen_dp_prob).to(self.device).apply(weights_init)
        self.net_D = Discriminator(disc_input_channel).to(self.device).apply(weights_init)
        self.net_G_opt = torch.optim.Adam(self.net_G.parameters(), lr=learning_rate)
        self.net_D_opt = torch.optim.Adam(self.net_D.parameters(), lr=learning_rate)
        
        # 损失函数初始化
        self.adv_criterion = nn.BCEWithLogitsLoss()
        # self.recon_criterion = nn.L1Loss()
        self.recon_criterion = nn.SmoothL1Loss()
        
        
    def run_with_mask(self):
        iter_counter=0
        for epoch_idx in tqdm(range(self.total_epoch),unit="epoch"):
            for ground_truth, mask in self.data_loader:
                # 初始化输入
                masked_data = ground_truth*mask
                
                mask = mask.to(self.device)
                ground_truth=ground_truth.to(self.device)
                masked_data=masked_data.to(self.device)
                
                # 首先更新鉴别器
                with torch.no_grad():
                    # 在不记录梯度的情况下走一遍生成器
                    fake = self.net_G(masked_data,mask)
                
                # 计算鉴别器损失
                D_fake_hat = self.net_D(fake.detach(),masked_data) # Detach generator
                D_fake_loss = self.adv_criterion(D_fake_hat, torch.zeros_like(D_fake_hat))
                D_real_hat = self.net_D(ground_truth, masked_data)
                D_real_loss = self.adv_criterion(D_real_hat, torch.ones_like(D_real_hat))
                D_loss = (D_fake_loss + D_real_loss) / 2
                
                # 对鉴别器反向传播
                self.net_D_opt.zero_grad() # Zero out the gradient before backpropagation
                D_loss.backward(retain_graph=True) # Update gradients
                self.net_D_opt.step() # Update optimizer

                # 更新生成器
                fake = self.net_G(masked_data,mask)
                disc_fake_hat = self.net_D(fake, masked_data)
                G_adv_loss = self.adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
                G_rec_loss = self.recon_criterion(ground_truth, fake)
                G_hole_loss = self.recon_criterion((1 - mask) * ground_truth, (1 - mask) * fake)
                G_vali_loss = self.recon_criterion(mask * ground_truth, mask * fake)
                
                G_loss = G_adv_loss + lambda_recon * G_rec_loss+G_hole_loss+G_vali_loss
                
                # 对生成器反向传播
                self.net_G_opt.zero_grad()
                G_loss.backward() # Update gradients
                self.net_G_opt.step() # Update optimizer

                # 保存和输出
                if (iter_counter+1) % self.display_step == 0 or iter_counter == 1:
                    if save_model:
                        file_name = f"{self.model_name}_{datetime.datetime.now().strftime('%m%d')}_{epoch_idx}epoch_{iter_counter}iter.pth"
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
                            file_name = f"{item_name}_{datetime.datetime.now().strftime('%m%d')}_{epoch_idx}epoch_{iter_counter}iter.raw"
                            file_path = os.path.join(data_save_path,"output",file_name)
                            raw_file = variable_list[item_name][0].cpu()
                            raw_file = raw_file.detach().numpy()
                            raw_file.astype('float32').tofile(file_path)
                
                if iter_counter%200==0:
                    wandb.log({"D_loss":D_loss,
                                "G_loss":G_loss,
                                "G_adv_loss":G_adv_loss,
                                "G_rec_loss":G_rec_loss,
                                "G_hole_loss":G_hole_loss,
                                "G_vali_loss":G_vali_loss
                                })
                    
                iter_counter += 1
        wandb.finish()
                
        
    def run(self):
        iter_counter=0
        for epoch_idx in tqdm(range(self.total_epoch),unit="epoch"):
            for ground_truth, mask in self.data_loader:
                # 初始化输入
                masked_data = ground_truth*mask
                ground_truth=ground_truth.to(self.device)
                masked_data=masked_data.to(self.device)
                
                # 首先更新鉴别器
                with torch.no_grad():
                    # 在不记录梯度的情况下走一遍生成器
                    fake = self.net_G(masked_data)
                
                # 计算鉴别器损失
                D_fake_hat = self.net_D(fake.detach(),masked_data) # Detach generator
                D_fake_loss = self.adv_criterion(D_fake_hat, torch.zeros_like(D_fake_hat))
                D_real_hat = self.net_D(ground_truth, masked_data)
                D_real_loss = self.adv_criterion(D_real_hat, torch.ones_like(D_real_hat))
                D_loss = (D_fake_loss + D_real_loss) / 2
                
                # 对鉴别器反向传播
                self.net_D_opt.zero_grad() # Zero out the gradient before backpropagation
                D_loss.backward(retain_graph=True) # Update gradients
                self.net_D_opt.step() # Update optimizer

                # 更新生成器
                fake = self.net_G(masked_data)
                disc_fake_hat = self.net_D(fake, masked_data)
                G_adv_loss = self.adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
                G_rec_loss = self.recon_criterion(ground_truth, fake)
                G_loss = G_adv_loss + lambda_recon * G_rec_loss
                
                # 对生成器反向传播
                self.net_G_opt.zero_grad()
                G_loss.backward() # Update gradients
                self.net_G_opt.step() # Update optimizer

                # 保存和输出
                if (iter_counter+1) % self.display_step == 0 or iter_counter == 1:
                    if save_model:
                        file_name = f"{self.model_name}_{datetime.datetime.now().strftime('%m%d')}_{epoch_idx}epoch_{iter_counter}iter.pth"
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
                            file_name = f"{item_name}_{datetime.datetime.now().strftime('%m%d')}_{epoch_idx}epoch_{iter_counter}iter.raw"
                            file_path = os.path.join(data_save_path,"output_data",file_name)
                            raw_file = variable_list[item_name][0].cpu()
                            raw_file = raw_file.detach().numpy()
                            raw_file.astype('float32').tofile(file_path)
                            
                    if self.cfg.WANDB.WORK:
                        if iter_counter%self.train.log_save_iter==0:
                            wandb.log({"D_loss":D_loss,
                                        "G_loss":G_loss,
                                        "G_adv_loss":G_adv_loss,
                                        "G_rec_loss":G_rec_loss,
                                        })
                    
                iter_counter += 1
        wandb.finish()
                
                
        




if __name__ == '__main__':
    trainer = DCGAN_Trainer(cfg)
    trainer.run()
    # trainer.run_with_mask()