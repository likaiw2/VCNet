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

import time


class GAN_Trainer:
    def __init__(self, cfg,net_G=InpaintSANet,net_D=InpaintSADirciminator):
        self.opt=cfg
        self.model_name = f"{self.opt.RUN.MODEL}"
        info = f" [Step: {self.num_step}/{self.opt.TRAIN.NUM_TOTAL_STEP} ({100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP}%)] "
        print(info)
        
        # 设置数据集
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
        
        # 加载模型参数
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
        for epoch in range(self.epoch_total):
            #validate(netG, netD, gan_loss, recon_loss, dis_loss, optG, optD, val_loader, i, device=cuda0)

            #train data
            self.train(netG=self.net_G, 
                       netD=self.net_D, 
                       gan_loss=self.gan_loss, 
                       recon_loss=self.recon_loss, 
                       dis_loss=self.dis_loss, 
                       optG=self.net_G_opt, 
                       optD=self.net_D_opt, 
                       train_loader=self.data_loader, 
                       epoch=epoch, 
                       device=self.device)

            if self.opt.RUN.SAVE_PTH:
                # if (global_iter + 1) % self.interval_save == 0 or (global_iter + 1) == self.interval_total or global_iter==0:
                if epoch%200==0 :
                    # save weights
                    fileName = f"{self.pth_save_path}/{self.model_name}_{epoch}epoch.pth"
                    os.makedirs(self.pth_save_path) if not os.path.exists(self.pth_save_path) else None
                    
                    torch.save({'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),}, fileName)
                    
                    # save images
                    tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                fileName=f"ground_truth",
                                volume=imgs)
                    tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                fileName=f"mask",
                                volume=mask)
                    tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                fileName=f"input",
                                volume=input)
                    tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                fileName=f"output",
                                volume=output)
        
    def train(netG, netD, GANLoss, ReconLoss, DLoss, optG, optD, dataloader, epoch, device):
        """
        Train Phase, for training and spectral normalization patch gan in
        Free-Form Image Inpainting with Gated Convolution (snpgan)

        """
        # init
        batch_time = tools.AverageMeter()
        data_time = tools.AverageMeter()
        losses = {"g_loss":tools.AverageMeter(), 
                  "r_loss":tools.AverageMeter(), 
                  "whole_loss":tools.AverageMeter(), 
                  'd_loss':tools.AverageMeter()}
        end = time.time()

        # set train mode
        netG.train()
        netD.train()
        
        # start train
        for i, (imgs, masks) in enumerate(dataloader):
            data_time.update(time.time() - end)
            masks = masks['random_free_form']

            # Optimize Discriminator
            optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()

            imgs, masks = imgs.to(device), masks.to(device)
            imgs = (imgs / 127.5 - 1)
            # mask is 1 on masked region

            coarse_imgs, recon_imgs, attention = netG(imgs, masks)
            #print(attention.size(), )
            complete_imgs = recon_imgs * masks + imgs * (1 - masks)

            pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
            neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
            pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

            pred_pos_neg = netD(pos_neg_imgs)
            pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
            d_loss = DLoss(pred_pos, pred_neg)
            losses['d_loss'].update(d_loss.item(), imgs.size(0))
            d_loss.backward(retain_graph=True)

            optD.step()


            # Optimize Generator
            optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
            pred_neg = netD(neg_imgs)
            #pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
            g_loss = GANLoss(pred_neg)
            r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)

            whole_loss = g_loss + r_loss

            # Update the recorder for losses
            losses['g_loss'].update(g_loss.item(), imgs.size(0))
            losses['r_loss'].update(r_loss.item(), imgs.size(0))
            losses['whole_loss'].update(whole_loss.item(), imgs.size(0))
            whole_loss.backward()

            optG.step()


            # Update time recorder
            batch_time.update(time.time() - end)

            if (i+1) % config.SUMMARY_FREQ == 0:
                # Logger logging
                logger.info("Epoch {0}, [{1}/{2}]: Batch Time:{batch_time.val:.4f},\t Data Time:{data_time.val:.4f}, Whole Gen Loss:{whole_loss.val:.4f}\t,"
                            "Recon Loss:{r_loss.val:.4f},\t GAN Loss:{g_loss.val:.4f},\t D Loss:{d_loss.val:.4f}" \
                            .format(epoch, i+1, len(dataloader), batch_time=batch_time, data_time=data_time, whole_loss=losses['whole_loss'], r_loss=losses['r_loss'] \
                            ,g_loss=losses['g_loss'], d_loss=losses['d_loss']))
                # Tensorboard logger for scaler and images
                info_terms = {'WGLoss':whole_loss.item(), 'ReconLoss':r_loss.item(), "GANLoss":g_loss.item(), "DLoss":d_loss.item()}

                for tag, value in info_terms.items():
                    tensorboardlogger.scalar_summary(tag, value, epoch*len(dataloader)+i)

                for tag, value in losses.items():
                    tensorboardlogger.scalar_summary('avg_'+tag, value.avg, epoch*len(dataloader)+i)

                def img2photo(imgs):
                    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()
                # info = { 'train/ori_imgs':img2photo(imgs),
                #          'train/coarse_imgs':img2photo(coarse_imgs),
                #          'train/recon_imgs':img2photo(recon_imgs),
                #          'train/comp_imgs':img2photo(complete_imgs),
                info = {
                        'train/whole_imgs':img2photo(torch.cat([ imgs * (1 - masks), coarse_imgs, recon_imgs, imgs, complete_imgs], dim=3))
                        }

                for tag, images in info.items():
                    tensorboardlogger.image_summary(tag, images, epoch*len(dataloader)+i)
            if (i+1) % config.VAL_SUMMARY_FREQ == 0 and val_datas is not None:

                validate(netG, netD, GANLoss, ReconLoss, DLoss, optG, optD, val_datas , epoch, device, batch_n=i)
                netG.train()
                netD.train()
            end = time.time()
        

        
        



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
                
                
                