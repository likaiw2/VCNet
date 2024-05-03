import os
import wandb
import torch
from tqdm import tqdm
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import matplotlib.pyplot as plt

from model.models import *
from configs.config import get_cfg_defaults  # 导入获取默认配置的函数
import utils.tools as tools
import utils.losses as losses


class GAN_Trainer:
    def __init__(self, cfg,net_G=InpaintSANet,net_D=InpaintSADirciminator):
        self.opt=cfg
        self.model_name = f"{self.opt.RUN.MODEL}"
        info = f" [Step: {self.num_step}/{self.opt.TRAIN.NUM_TOTAL_STEP} ({100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP}%)] "
        print(info)
        
        self.dataset = tools.DataSet(data_path = self.opt.PATH.DATA_PATH,
                                     volume_shape = self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape = self.opt.DATASET.TARGET_SHAPE,
                                     mask_type = self.opt.RUN.TYPE,
                                     data_type = np.float32)
        
        self.data_loader = data.DataLoader(dataset=self.dataset, 
                                       batch_size=self.opt.TRAIN.BATCH_SIZE, 
                                       shuffle=self.opt.DATASET.SHUFFLE, 
                                       num_workers=self.opt.SYSTEM.NUM_WORKERS)
        self.data_iter = iter(self.data_loader)
                          
        # 初始化wandb，用于实验跟踪和可视化
        # self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        # cfg.freeze()
        
        # self.wandb = wandb
        # self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, 
        #                 resume=self.opt.TRAIN.RESUME, 
        #                 notes=self.opt.WANDB.LOG_DIR, 
        #                 config=self.opt,
        #                 mode=self.opt.WANDB.MODE)
        
        
        
        self.device=torch.device(self.opt.SYSTEM.DEVICE)
        self.epoch_total=self.opt.TRAIN.EPOCH_TOTAL
        self.interval_start=self.opt.TRAIN.INTERVAL_START
        self.interval_save=self.opt.TRAIN.INTERVAL_SAVE
        self.save_path=self.opt.PATH.SAVE_PATH
        self.pth_save_path=self.opt.PATH.PTH_SAVE_PATH
        self.interval_total=self.epoch_total*len(self.dataset)
        print("total iter: ",self.interval_total)

        # 创建模型组件
        self.net_G = net_G().to(self.device)
        self.net_D = net_D().to(self.device)
        self.net_G_opt = torch.optim.Adam(net_G.parameters(), lr=lr, weight_decay=decay)
        self.net_D_opt = torch.optim.Adam(net_D.parameters(), lr=lr, weight_decay=decay)
        
        if self.opt.RUN.LOAD_PTH:
            if os.path.exists(self.opt.PATH.PTH_LOAD_PATH):
                loaded_state = torch.load(self.opt.PATH.PTH_LOAD_PATH)
                self.model.load_state_dict(loaded_state["net_G"])
                self.model.load_state_dict(loaded_state["net_D"])
                self.optimizer.load_state_dict(loaded_state["net_G_opt"])
                self.optimizer.load_state_dict(loaded_state["net_D_opt"])
                print("Weight load success!")
            else:
                print("load weights failed!")

        # 初始化训练步数
        self.num_step = self.opt.TRAIN.START_STEP

        # 定义损失函数
        self.recon_loss = losses.ReconLoss(*(config.L1_LOSS_ALPHA))
        self.gan_loss = losses.SNGenLoss(config.GAN_LOSS_ALPHA)
        self.dis_loss = losses.SNDisLoss()
        
    def run(self):
        global_iter=0
        for epoch_idx in tqdm(range(self.epoch_total)):
            for imgs, masks in self.dataloader:

                # Optimize Discriminator
                self.net_G.zero_grad()
                self.net_D.zero_grad()
                self.net_G_opt.zero_grad()
                self.net_D_opt.zero_grad()

                imgs, masks = imgs.to(self.device), masks.to(self.device)
                
                coarse_imgs, recon_imgs = self.net_G(imgs, masks)
                
                complete_imgs = recon_imgs * masks + imgs * masks # mask is 0 on masked region

                pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
                neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
                pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

                pred_pos_neg = self.net_D(pos_neg_imgs)
                pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
                d_loss = self.dis_loss(pred_pos, pred_neg)
                losses['d_loss'].update(d_loss.item(), imgs.size(0))
                d_loss.backward(retain_graph=True)

                self.net_D_opt.step()


                # Optimize Generator
                self.net_G.zero_grad()
                self.net_D.zero_grad()
                self.net_G_opt.zero_grad()
                self.net_D_opt.zero_grad()
                
                pred_neg = self.net_D(neg_imgs)
                #pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
                g_loss = self.gan_loss(pred_neg)
                r_loss = self.recon_loss(imgs, coarse_imgs, recon_imgs, masks)

                whole_loss = g_loss + r_loss

                # Update the recorder for losses
                losses['g_loss'].update(g_loss.item(), imgs.size(0))
                losses['r_loss'].update(r_loss.item(), imgs.size(0))
                losses['whole_loss'].update(whole_loss.item(), imgs.size(0))
                whole_loss.backward()

                self.net_G_opt.step()

                if self.opt.RUN.SAVE_PTH:
                    if (global_iter + 1) % self.interval_save == 0 or (global_iter + 1) == self.interval_total or global_iter==0:
                        # save weights
                        fileName = f"{self.pth_save_path}/{self.model_name}_{global_iter+1}iter.pth"
                        os.makedirs(self.pth_save_path) if not os.path.exists(self.pth_save_path) else None
                        
                        torch.save({'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),}, fileName)
                        
                        # save images
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"ground_truth",
                                    volume=imgs)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"mask",
                                    volume=mask)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"input",
                                    volume=input)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"output",
                                    volume=output)
                    

            saved_model = {
                'epoch': i + 1,
                'netG_state_dict': netG.to(cpu0).state_dict(),
                'netD_state_dict': netD.to(cpu0).state_dict(),
                # 'optG' : optG.state_dict(),
                # 'optD' : optD.state_dict()
            }
            torch.save(saved_model, '{}/epoch_{}_ckpt.pth.tar'.format(log_dir, i+1))
            torch.save(saved_model, '{}/latest_ckpt.pth.tar'.format(log_dir, i+1))
        



class UnetTrainer:
    def __init__(self,cfg,model=PConvUNet()):
        self.opt=cfg
        self.model_name = f"{self.opt.RUN.MODEL}"
        
        
        self.dataset = tools.DataSet(data_path = self.opt.PATH.DATA_PATH,
                                     volume_shape = self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape = self.opt.DATASET.TARGET_SHAPE,
                                     mask_type = self.opt.RUN.TYPE,
                                     data_type = np.float32)
        
        self.data_loader = data.DataLoader(dataset=self.dataset, 
                                       batch_size=self.opt.TRAIN.BATCH_SIZE, 
                                       shuffle=self.opt.DATASET.SHUFFLE, 
                                       num_workers=self.opt.SYSTEM.NUM_WORKERS)
        self.data_iter = iter(self.data_loader)
        
        
        self.device=torch.device(self.opt.SYSTEM.DEVICE)
        self.epoch_total=self.opt.TRAIN.EPOCH_TOTAL
        self.interval_start=self.opt.TRAIN.INTERVAL_START
        self.interval_save=self.opt.TRAIN.INTERVAL_SAVE
        self.save_path=self.opt.PATH.SAVE_PATH
        self.pth_save_path=self.opt.PATH.PTH_SAVE_PATH
        self.interval_total=self.epoch_total*len(self.dataset)
        print("total iter: ",self.interval_total)
        
        
        
        # 初始化模型和优化器
        self.model=model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.TRAIN.LEARN_RATE)
        if self.opt.RUN.LOAD_PTH:
            if os.path.exists(self.opt.PATH.PTH_LOAD_PATH):
                loaded_state = torch.load(self.opt.PATH.PTH_LOAD_PATH)
                self.model.load_state_dict(loaded_state["model"])
                self.optimizer.load_state_dict(loaded_state["optimizer"])
                print("Weight load success!")
            else:
                print("load weights failed!")
        
        
        
        # self.loss_function = losses.InpaintingLoss(tools.VGG16FeatureExtractor(pth_path=self.opt.PATH.VGG16_PATH)).to(self.device)
        self.loss_function = losses.WMSELoss()
        
        
        
    def run(self):
        global_iter=0
        for epoch_idx in tqdm(range(self.epoch_total)):
            for gt,mask in self.data_loader:
                
                gt = gt.to(self.device)
                mask = mask.to(self.device)

                input = gt*mask
                output, _ = self.model(input, mask)
                loss_dict = self.loss_function(mask, output, gt)
                loss = loss_dict
                
                # 加权计算并输出损失
                # loss = 0.0
                # lambda_dict = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
                # for key, coef in lambda_dict.items():
                #     value = coef * loss_dict[key]
                #     loss += value
                    # if (i + 1) % args.log_interval == 0:
                    #     writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.opt.RUN.SAVE_PTH:
                    if (global_iter + 1) % self.interval_save == 0 or (global_iter + 1) == self.interval_total or global_iter==0:
                        # save weights
                        fileName = f"{self.pth_save_path}/{self.model_name}_{global_iter+1}iter.pth"
                        os.makedirs(self.pth_save_path) if not os.path.exists(self.pth_save_path) else None
                        
                        torch.save({'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),}, fileName)
                        
                        # save images
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                    fileName=f"ground_truth",
                                    volume=gt)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                    fileName=f"mask",
                                    volume=mask)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                    fileName=f"input",
                                    volume=input)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                    fileName=f"output",
                                    volume=output)
                    
                global_iter+=1
                

if __name__ == '__main__':
    trainer=UnetTrainer(cfg)
    loader = trainer.data_loader
    imgs, mask= next(iter(loader))
    print(imgs.shape)
    print(mask.shape)
                
                
                