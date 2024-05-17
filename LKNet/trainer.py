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


class SAGAN_Trainer:
    def __init__(self, cfg, net_G=InpaintSANet, net_D=InpaintSADirciminator):
        self.opt = cfg
        self.model_name = f"{self.opt.RUN.MODEL}{self.opt.RUN.ADD_INFO}"
        # info = f" [Step: {self.num_step}/{self.opt.TRAIN.NUM_TOTAL_STEP} ({100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP}%)] "
        # print(info)

        # 设置数据集
        self.dataset = tools.DataSet(data_path=self.opt.PATH.DATA_PATH,
                                     volume_shape=self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape=self.opt.DATASET.TARGET_SHAPE,
                                     mask_type=self.opt.RUN.TYPE,
                                     data_type=np.float32)

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

        self.device = torch.device(self.opt.SYSTEM.DEVICE)
        
        self.interval_start = self.opt.TRAIN.INTERVAL_START
        self.interval_save = self.opt.TRAIN.INTERVAL_SAVE
        
        self.save_path = self.opt.PATH.SAVE_PATH
        self.pth_save_path = self.opt.PATH.PTH_SAVE_PATH
        self.interval_total = self.opt.TRAIN.EPOCH_TOTAL*len(self.dataset)
        print("total iter: ", self.interval_total)

        # 创建模型组件
        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)
        
        self.net_G_opt = torch.optim.Adam(
            net_G.parameters(), lr=self.opt.TRAIN.LEARN_RATE, weight_decay=0.0)
        self.net_D_opt = torch.optim.Adam(
            net_D.parameters(), lr=self.opt.TRAIN.LEARN_RATE, weight_decay=0.0)

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

        # 定义损失函数
        self.recon_loss = losses.ReconLoss(*([1.2, 1.2, 1.2, 1.2]))
        self.gan_loss = losses.SNGenLoss(0.005)
        self.dis_loss = losses.SNDisLoss()

    def run(self):
        netG = self.net_G
        netD = self.net_D
        GANLoss = self.gan_loss
        ReconLoss=self.recon_loss
        DLoss = self.dis_loss
        optG = self.net_G_opt
        optD = self.net_D_opt
        dataloader = self.data_loader
        device = self.device
        
        global_iter = 0
        
        for epoch in tqdm(range(self.opt.TRAIN.INTERVAL_START,self.opt.TRAIN.EPOCH_TOTAL)):
            """
            Train Phase, for training and spectral normalization patch gan in
            Free-Form Image Inpainting with Gated Convolution (snpgan)
            """
            # init
            end = time.time()
            batch_time = tools.AverageMeter()
            data_time = tools.AverageMeter()
            losses = {"g_loss": tools.AverageMeter(),
                    "r_loss": tools.AverageMeter(),
                    "whole_loss": tools.AverageMeter(),
                    'd_loss': tools.AverageMeter()}

            # set train mode
            netG.train()
            netD.train()

            # start train
            for i, (imgs, masks) in enumerate(dataloader):
                data_time.update(time.time() - end)

                # Optimize Discriminator
                optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()

                imgs = imgs.to(device)
                masks = masks.to(device)
                
                # 粗糙的imgs和精修的imgs
                coarse_imgs, recon_imgs = netG(imgs, masks)
                # 制作补全后的图像，非挖空区域是原图，挖空区域是补全后的图片
                complete_imgs = recon_imgs * (1-masks) + imgs * masks   # mask is 0 on masked regiona

                # 制作正面图像和负面图像并把它们合并到一起，送进鉴别器
                pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
                neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
                pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

                # 鉴别器执行操作
                pred_pos_neg = netD(pos_neg_imgs)
                pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)    # 分别读取鉴别器对正面样本的反应和负面样本的反应
                # 求损失并进行反向传播
                d_loss = DLoss(pred_pos, pred_neg)
                losses['d_loss'].update(d_loss.item(), imgs.size(0))
                d_loss.backward(retain_graph=True)

                optD.step()

                # Optimize Generator
                optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
                pred_neg = netD(neg_imgs)
                # pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
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
                end = time.time()
                
                # if self.num_step % self.opt.TRAIN.VISUALIZE_INTERVAL == 0:
                #     idx = self.opt.WANDB.NUM_ROW
                #     self.wandb.log({"examples": [
                #         self.wandb.Image(self.to_pil(y_imgs[idx].cpu()), caption="original_image"),
                #         self.wandb.Image(self.to_pil(linear_unscaling(cont_imgs[idx]).cpu()), caption="contaminant_image"),
                #         self.wandb.Image(self.to_pil(linear_unscaling(masked_imgs[idx]).cpu()), caption="masked_image"),
                #         self.wandb.Image(self.to_pil(masks[idx].cpu()), caption="original_masks"),
                #         self.wandb.Image(self.to_pil(smooth_masks[idx].cpu()), caption="smoothed_masks"),
                #         self.wandb.Image(self.to_pil(pred_masks[idx].cpu()), caption="predicted_masks"),
                #         self.wandb.Image(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()), caption="output")
                #     ]}, commit=False)
                # self.wandb.log({})
                
                if self.opt.RUN.SAVE_PTH:
                    if (global_iter + 1) % self.interval_save == 0 or (global_iter + 1) == self.interval_total or global_iter==0:
                    # if epoch % 200 == 0:
                        # save weights
                        fileName = f"{self.pth_save_path}/{self.model_name}_{epoch}epoch.pth"
                        os.makedirs(self.pth_save_path) if not os.path.exists(
                            self.pth_save_path) else None

                        torch.save({'net_G': self.net_G.state_dict(),
                                    'net_D': self.net_D.state_dict(),
                                    'net_G_opt': self.net_G_opt.state_dict(), 
                                    'net_D_opt': self.net_D_opt.state_dict(), 
                                    }, fileName)

                        # save images
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                    fileName=f"ground_truth",
                                    volume=imgs)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                    fileName=f"masks",
                                    volume=masks)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                    fileName=f"coarse_imgs",
                                    volume=coarse_imgs)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                    fileName=f"recon_imgs",
                                    volume=recon_imgs)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch}epoch",
                                    fileName=f"complete_imgs",
                                    volume=complete_imgs)
                global_iter +=1
            
            
# class P2P_Trainer:
#     def __init__(self,cfg,net_G, net_D) -> None:
#         self.opt = cfg
#         self.model_name = f"{self.opt.RUN.MODEL}"
#         # info = f" [Step: {self.num_step}/{self.opt.TRAIN.NUM_TOTAL_STEP} ({100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP}%)] "
#         # print(info)

#         # 设置数据集
#         self.dataset = tools.DataSet(data_path=self.opt.PATH.DATA_PATH,
#                                      volume_shape=self.opt.DATASET.ORIGIN_SHAPE,
#                                      target_shape=self.opt.DATASET.TARGET_SHAPE,
#                                      mask_type=self.opt.RUN.TYPE,
#                                      data_type=np.float32)
#         self.data_loader = data.DataLoader(dataset=self.dataset,
#                                            batch_size=self.opt.TRAIN.BATCH_SIZE,
#                                            shuffle=self.opt.DATASET.SHUFFLE,
#                                            num_workers=self.opt.SYSTEM.NUM_WORKERS)
#         self.data_iter = iter(self.data_loader)
        
#         # 初始化变量
#         self.device = torch.device(self.opt.SYSTEM.DEVICE)
#         self.interval_start = self.opt.TRAIN.INTERVAL_START
#         self.interval_save = self.opt.TRAIN.INTERVAL_SAVE
        
#         self.save_path = self.opt.PATH.SAVE_PATH
#         self.pth_save_path = self.opt.PATH.PTH_SAVE_PATH
#         self.interval_total = self.opt.TRAIN.EPOCH_TOTAL*len(self.dataset)
#         print("total iter: ", self.interval_total)
        
#         gen = p2pUNet(input_channels=1, output_channels=1).to(self.device)
#         gen_opt = torch.optim.Adam(gen.parameters(), lr=self.opt.)
#         disc = Discriminator(input_dim + real_dim).to(device)
#         disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
        
#     def run(self):
#         pass

class UnetTrainer:
    def __init__(self, cfg, model=PConvUNet()):
        self.opt = cfg
        self.model_name = f"{self.opt.RUN.MODEL}_{self.opt.RUN.ADDITIONAL_INFO}"

        self.dataset = tools.DataSet(data_path=self.opt.PATH.DATA_PATH,
                                     volume_shape=self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape=self.opt.DATASET.TARGET_SHAPE,
                                     mask_type=self.opt.RUN.TYPE,
                                     data_type=np.float32)

        self.data_loader = data.DataLoader(dataset=self.dataset,
                                           batch_size=self.opt.TRAIN.BATCH_SIZE,
                                           shuffle=self.opt.DATASET.SHUFFLE,
                                           num_workers=self.opt.SYSTEM.NUM_WORKERS)
        self.data_iter = iter(self.data_loader)

        self.device = torch.device(self.opt.SYSTEM.DEVICE)
        
        self.save_path = f"{self.opt.PATH.SAVE_PATH}/{self.model_name}"
        self.pth_save_path = f"{self.opt.PATH.PTH_SAVE_PATH}/{self.model_name}"
        
        self.interval_total = self.opt.TRAIN.EPOCH_TOTAL*len(self.dataset)

        # 初始化模型和优化器
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.WEIGHT.LEARN_RATE)
        if self.opt.RUN.LOAD_PTH:
            if os.path.exists(self.opt.PATH.PTH_LOAD_PATH):
                loaded_state = torch.load(self.opt.PATH.PTH_LOAD_PATH)
                self.model.load_state_dict(loaded_state["model"])
                self.optimizer.load_state_dict(loaded_state["optimizer"])
                print("Weight load success!")
            else:
                print("load weights failed!")

        self.loss_function = losses.InpaintingLoss3D().to(self.device)
        # self.loss_function = losses.WMSELoss().to(self.device)

    def run(self):
        global_iter = 0
        for epoch_idx in tqdm(range(self.opt.TRAIN.EPOCH_TOTAL),unit="epoch"):
            _loss = []
            for gt, mask in tqdm(self.data_loader,unit="iter",leave=False):
                gt = gt.to(self.device)
                mask = mask.to(self.device)

                input = gt*mask                                     # 制作输入图像
                output_raw, output_mask = self.model(input, mask)                 # 制作输出
                output_final=output_raw*(1-output_mask)+input*output_mask
                loss_dict = self.loss_function(input, mask, output_raw, gt)    # 求损失
                loss = loss_dict
                
                # 加权计算并输出损失
                loss = 0.0
                lambda_dict = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
                for key, coef in lambda_dict.items():
                    value = coef * loss_dict[key]
                    loss += value
                # if (i + 1) % args.log_interval == 0:
                #     writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
                _loss.append(loss.item())

                self.optimizer.zero_grad()          # 重置梯度
                loss.backward()                     # 计算梯度
                self.optimizer.step()               # 根据梯度和优化器的参数更新参数

                if self.opt.RUN.SAVE_PTH:
                    # 第一次迭代保存，最后一次迭代保存，中间的话看指定的保存间隔
                    if (global_iter + 1) % self.opt.TRAIN.ITER_SAVE == 0 or (global_iter + 1) == self.interval_total or global_iter == 0:
                        # save weights
                        fileName = f"{self.pth_save_path}/{self.model_name}_{global_iter+1}iter.pth"
                        os.makedirs(self.pth_save_path) if not os.path.exists(
                            self.pth_save_path) else None

                        torch.save({'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(), }, fileName)
                        for i in range(self.opt.TRAIN.BATCH_SIZE):
                            # save images
                            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                        fileName=f"b{i}_ground_truth",
                                        volume=gt[i])
                            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                        fileName=f"b{i}_mask",
                                        volume=mask[i])
                            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                        fileName=f"b{i}_input",
                                        volume=input[i])
                            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                        fileName=f"b{i}_output_raw",
                                        volume=output_raw[i])
                            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                        fileName=f"b{i}_output_final",
                                        volume=output_final[i])
                            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                                        fileName=f"b{i}_output_final",
                                        volume=output_mask[i])
                # loop.set_description(f'Epoch [{epoch_idx}/{self.opt.TRAIN.EPOCH_TOTAL}], Iter [{global_iter}/{self.interval_total}]\n')
                # loop.set_postfix(loss = loss.item())

                global_iter += 1
            average_loss=np.average(np.array(_loss))
            print("\n",average_loss)

if __name__ == '__main__':
    trainer = UnetTrainer(cfg)
    loader = trainer.data_loader
    imgs, mask = next(iter(loader))
    print(imgs.shape)
    print(mask.shape)
