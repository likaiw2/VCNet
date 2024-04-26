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
    def __init__(self, cfg):
        self.opt = cfg
        # 根据配置参数构造模型名称
        self.model_name = f"{self.opt.RUN.MODEL}_{self.opt.TRAIN.NUM_TOTAL_STEP}step"
                          
        # 初始化wandb，用于实验跟踪和可视化
        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        cfg.freeze()
        
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, 
                        resume=self.opt.TRAIN.RESUME, 
                        notes=self.opt.WANDB.LOG_DIR, 
                        config=self.opt,
                        mode=self.opt.WANDB.MODE)
        
        self.dataset = tools.DataSet(data_path = self.opt.PATH.SOURCE_PATH,
                                     volume_shape = self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape = self.opt.DATASET.TARGET_SHAPE,
                                     mask_type = self.opt.RUN.TYPE,
                                     data_type=self.opt.DATA_TYPE)

        # 创建一个图像数据加载器
        self.image_loader = data.DataLoader(dataset=self.dataset, 
                                       batch_size=self.opt.TRAIN.BATCH_SIZE, 
                                       shuffle=self.opt.DATASET.SHUFFLE, 
                                       num_workers=self.opt.SYSTEM.NUM_WORKERS)
        self.data_size = len(self.image_loader)

        self.saveRAW = tools.saveRAW()

        # 创建模型组件
        self.gen = PConvUNet()
        
        # TODO
        self.discriminator = Discriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)
        self.patch_discriminator = PatchDiscriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)

        # 创建优化器
        self.optimizer_mpn = torch.optim.Adam(self.mpn.parameters(), lr=self.opt.MODEL.MPN.LR, betas=self.opt.MODEL.MPN.BETAS)
        self.optimizer_rin = torch.optim.Adam(self.rin.parameters(), lr=self.opt.MODEL.RIN.LR, betas=self.opt.MODEL.RIN.BETAS)
        self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()) + list(self.patch_discriminator.parameters()), lr=self.opt.MODEL.D.LR, betas=self.opt.MODEL.D.BETAS)
        self.optimizer_joint = torch.optim.Adam(list(self.mpn.parameters()) + list(self.rin.parameters()), lr=self.opt.MODEL.JOINT.LR, betas=self.opt.MODEL.JOINT.BETAS)

        # 初始化训练步数
        self.num_step = self.opt.TRAIN.START_STEP

        # 如果指定了开始步数并且需要恢复训练，则加载检查点
        if self.opt.TRAIN.START_STEP != 0 and self.opt.TRAIN.RESUME:  # find start step from checkpoint file name. TODO
            log.info("Checkpoints loading...")
            self.load_checkpoints(self.opt.TRAIN.START_STEP)

        # 检查并使用多GPU
        self.check_and_use_multi_gpu()

        # 定义损失函数
        self.weighted_bce_loss = WeightedBCELoss().cuda()
        self.reconstruction_loss = torch.nn.L1Loss().cuda()
        self.semantic_consistency_loss = SemanticConsistencyLoss().cuda()
        self.texture_consistency_loss = IDMRFLoss().cuda()

    def run(self):
        # 循环直到达到总训练步数
        while self.num_step < self.opt.TRAIN.NUM_TOTAL_STEP:
            # 增加训练步数并打印进度信息
            self.num_step += 1
            info = f" [Step: {self.num_step}/{self.opt.TRAIN.NUM_TOTAL_STEP} ({100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP}%)] "
            print(info)

            # 从数据加载器中获取一批图像
            imgs, _ = next(iter(self.image_loader))
            y_imgs = imgs.float().cuda()
            imgs = linear_scaling(imgs.float().cuda())
            batch_size, channels, h, w = imgs.size()

            # 生成遮罩
            masks = torch.from_numpy(self.mask_generator.generate(h, w)).repeat([batch_size, 1, 1, 1]).float().cuda()

            # 从content数据加载器中获取一批图像
            cont_imgs = next(iter(self.cont_image_loader))
            cont_imgs = linear_scaling(cont_imgs.float().cuda())
            if cont_imgs.size(0) != imgs.size(0):
                cont_imgs = cont_imgs[:imgs.size(0)]

            # 应用平滑遮罩
            smooth_masks = self.mask_smoother(1 - masks) + masks
            smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

            # 根据遮罩合成图像
            masked_imgs = cont_imgs * smooth_masks + imgs * (1. - smooth_masks)
            self.unknown_pixel_ratio = torch.sum(masks.view(batch_size, -1), dim=1).mean() / (h * w)
            # 我想保存maskedpic 但是还没做 TODO
            # print(imgs.shape)
            # print(cont_imgs.shape)
            # print(masked_imgs.shape)
            # print(smooth_masks.shape)
            # print()
            
                        
            # 训练判别器
            for _ in range(self.opt.MODEL.D.NUM_CRITICS):
                d_loss = self.train_D(masked_imgs, masks, y_imgs)
            info += "D Loss: {} ".format(d_loss)

            # 训练生成器
            m_loss, g_loss, pred_masks, output = self.train_G(masked_imgs, masks, y_imgs)
            info += f"M Loss: {m_loss} G Loss: {g_loss} "
            
            # 记录日志
            if self.num_step % self.opt.TRAIN.LOG_INTERVAL == 0:
                log.info(info)

            # 记录可视化结果
            if self.num_step % self.opt.TRAIN.VISUALIZE_INTERVAL == 0:
                idx = self.opt.WANDB.NUM_ROW
                self.wandb.log({"examples": [
                    self.wandb.Image(self.to_pil(y_imgs[idx].cpu()), caption="original_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(cont_imgs[idx]).cpu()), caption="contaminant_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(masked_imgs[idx]).cpu()), caption="masked_image"),
                    self.wandb.Image(self.to_pil(masks[idx].cpu()), caption="original_masks"),
                    self.wandb.Image(self.to_pil(smooth_masks[idx].cpu()), caption="smoothed_masks"),
                    self.wandb.Image(self.to_pil(pred_masks[idx].cpu()), caption="predicted_masks"),
                    self.wandb.Image(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()), caption="output")
                ]}, commit=False)
            self.wandb.log({})
            # 保存检查点
            if (self.num_step % self.opt.TRAIN.SAVE_INTERVAL == 0 and self.num_step != 0) or self.num_step==1:
                self.do_checkpoint(self.num_step,imgs,cont_imgs,masked_imgs,smooth_masks)

    def train_D(self, x, y_masks, y):
        '''
        x -         残缺的图片
        y_masks -   真正的mask
        y -         原图
        '''
        # 清除判别器的梯度
        self.optimizer_discriminator.zero_grad()

        # 生成预测遮罩和颈部特征
        pred_masks, neck = self.mpn(x)
        # 生成输出
        output = self.rin(x, pred_masks, neck)

        # 计算真实和伪造的全局有效性
        real_global_validity = self.discriminator(y).mean()
        fake_global_validity = self.discriminator(output.detach()).mean()
        # 计算梯度惩罚
        gp_global = compute_gradient_penalty(self.discriminator, output.data, y.data)

        # 计算真实和伪造的局部有效性
        real_patch_validity = self.patch_discriminator(y, y_masks).mean()
        fake_patch_validity = self.patch_discriminator(output.detach(), y_masks).mean()
        gp_fake = compute_gradient_penalty(self.patch_discriminator, output.data, y.data, y_masks)

        # 计算真实和伪造的有效性
        real_validity = real_global_validity + real_patch_validity
        fake_validity = fake_global_validity + fake_patch_validity
        gp = gp_global + gp_fake

        # 计算判别器的损失
        d_loss = -real_validity + fake_validity + self.opt.OPTIM.GP * gp
        d_loss.backward()
        # 更新判别器的参数
        self.optimizer_discriminator.step()

        # 记录判别器的训练信息
        self.wandb.log({"real_global_validity": -real_global_validity.item(),
                        "fake_global_validity": fake_global_validity.item(),
                        "real_patch_validity": -real_patch_validity.item(),
                        "fake_patch_validity": fake_patch_validity.item(),
                        "gp_global": gp_global.item(),
                        "gp_fake": gp_fake.item(),
                        "real_validity": -real_validity.item(),
                        "fake_validity": fake_validity.item(),
                        "gp": gp.item()}, commit=False)
        return d_loss.item()

    def train_G(self, x, y_masks, y):
        # 根据训练步数选择优化器
        if self.num_step < self.opt.TRAIN.NUM_STEPS_FOR_JOINT:
            # 使用单独的优化器
            self.optimizer_mpn.zero_grad()
            self.optimizer_rin.zero_grad()

            # 生成预测遮罩
            pred_masks, neck = self.mpn(x)
            # 计算遮罩损失
            m_loss = self.weighted_bce_loss(pred_masks, y_masks, torch.tensor([1 - self.unknown_pixel_ratio, self.unknown_pixel_ratio]))
            # 记录遮罩损失
            self.wandb.log({"m_loss": m_loss.item()}, commit=False)
            # 更新遮罩生成网络的参数
            m_loss = self.opt.OPTIM.MASK * m_loss
            m_loss.backward(retain_graph=True)
            self.optimizer_mpn.step()
            # 根据遮罩和颈部特征生成输出
            if self.opt.MODEL.RIN.EMBRACE:
                x_embraced = x.detach() * (1 - pred_masks.detach())
                output = self.rin(x_embraced, pred_masks.detach(), neck.detach())
            else:
                output = self.rin(x, pred_masks.detach(), neck.detach())
            # 计算重建损失、语义一致性损失和纹理一致性损失
            recon_loss = self.reconstruction_loss(output, y)
            sem_const_loss = self.semantic_consistency_loss(output, y)
            tex_const_loss = self.texture_consistency_loss(output, y)
            # 计算全局和局部的对抗损失
            adv_global_loss = -self.discriminator(output).mean()
            adv_patch_loss = -self.patch_discriminator(output, y_masks).mean()
            adv_loss = adv_global_loss + adv_patch_loss

            # 计算生成器的总损失
            g_loss = self.opt.OPTIM.RECON * recon_loss + \
                     self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                     self.opt.OPTIM.TEXTURE * tex_const_loss * \
                     self.opt.OPTIM.ADVERSARIAL * adv_loss
            # 更新生成器的参数
            g_loss.backward()
            self.optimizer_rin.step()
        else:
            # 使用联合优化器
            self.optimizer_joint.zero_grad()
            pred_masks, neck = self.mpn(x)
            m_loss = self.weighted_bce_loss(pred_masks, y_masks, torch.tensor([1 - self.unknown_pixel_ratio, self.unknown_pixel_ratio]))
            self.wandb.log({"m_loss": m_loss.item()}, commit=False)
            m_loss = self.opt.OPTIM.MASK * m_loss
            if self.opt.MODEL.RIN.EMBRACE:
                x_embraced = x.detach() * (1 - pred_masks.detach())
                output = self.rin(x_embraced, pred_masks.detach(), neck.detach())
            else:
                output = self.rin(x, pred_masks.detach(), neck.detach())
            recon_loss = self.reconstruction_loss(output, y)
            sem_const_loss = self.semantic_consistency_loss(output, y)
            tex_const_loss = self.texture_consistency_loss(output, y)
            adv_global_loss = -self.discriminator(output).mean()
            adv_patch_loss = -self.patch_discriminator(output, y_masks).mean()
            adv_loss = adv_global_loss + adv_patch_loss

            g_loss = self.opt.OPTIM.RECON * recon_loss + \
                     self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                     self.opt.OPTIM.TEXTURE * tex_const_loss + \
                     self.opt.OPTIM.ADVERSARIAL * adv_loss

            final_loss = self.opt.MODEL.MPN.LOSS_COEFF * m_loss + self.opt.MODEL.RIN.LOSS_COEFF * g_loss
            final_loss.backward()
            self.optimizer_joint.step()
        # 记录生成器的训练信息
        self.wandb.log({"recon_loss": recon_loss.item(),
                        "sem_const_loss": sem_const_loss.item(),
                        "tex_const_loss": tex_const_loss.item(),
                        "adv_global_loss": adv_global_loss.item(),
                        "adv_patch_loss": adv_patch_loss.item(),
                        "adv_loss": adv_loss.item()}, commit=False)
        # 返回损失值和生成的遮罩和输出
        return m_loss.item(), g_loss.item(), pred_masks.detach(), output.detach()

    def check_and_use_multi_gpu(self):
        # 如果有多个GPU并且配置了多个GPU，则使用所有GPU
        if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
            log.info("Using {} GPUs...".format(torch.cuda.device_count()))
            self.mpn = torch.nn.DataParallel(self.mpn).cuda()
            self.rin = torch.nn.DataParallel(self.rin).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            self.patch_discriminator = torch.nn.DataParallel(self.patch_discriminator).cuda()
            self.mask_smoother = torch.nn.DataParallel(self.mask_smoother).cuda()
        else:
            log.info("GPU ID: {}".format(torch.cuda.current_device()))
            self.mpn = self.mpn.cuda()
            self.rin = self.rin.cuda()
            self.discriminator = self.discriminator.cuda()
            self.patch_discriminator = self.patch_discriminator.cuda()
            self.mask_smoother = self.mask_smoother.cuda()

    def do_checkpoint(self, num_step,imgs,cont_imgs,masked_imgs,smooth_masks):
        # 创建保存检查点的目录
        if not os.path.exists("./{}/{}/checkpoint-{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step)):
            os.makedirs("./{}/{}/checkpoint-{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name,num_step), exist_ok=True)

        # 创建检查点字典
        checkpoint = {
            'num_step': num_step,
            'mpn': self.mpn.state_dict(),
            'rin': self.rin.state_dict(),
            'D': self.discriminator.state_dict(),
            'patch_D': self.patch_discriminator.state_dict(),
            'optimizer_mpn': self.optimizer_mpn.state_dict(),
            'optimizer_rin': self.optimizer_rin.state_dict(),
            'optimizer_joint': self.optimizer_joint.state_dict(),
            'optimizer_D': self.optimizer_discriminator.state_dict(),
            # 'scheduler_mpn': self.scheduler_mpn.state_dict(),
            # 'scheduler_rin': self.scheduler_rin.state_dict(),
            # 'scheduler_joint': self.scheduler_joint.state_dict(),
            # 'scheduler_D': self.scheduler_discriminator.state_dict(),
        }
        torch.save(checkpoint, "./{}/{}/checkpoint-{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step,num_step))

        # 保存图片作参考
        for i in range(self.opt.TRAIN.BATCH_SIZE):
            save_path = f"./{self.opt.TRAIN.SAVE_DIR}/{self.model_name}/checkpoint-{num_step}"
            current_batch=int(num_step//(self.data_size/self.opt.TRAIN.BATCH_SIZE))
            
            save_image(imgs[i],f"{save_path}/epoch{current_batch}_batch_{i}_gt.jpg")
            save_image(cont_imgs[i],f"{save_path}/epoch{current_batch}_batch_{i}_cont_img.jpg")
            save_image(masked_imgs[i],f"{save_path}/epoch{current_batch}_batch_{i}_masked_img.jpg")
            save_image(smooth_masks[i],f"{save_path}/epoch{current_batch}_batch_{i}_smooth_masks.jpg")
            
            
    def load_checkpoints(self, num_step):
        checkpoints = torch.load("./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))
        self.num_step = checkpoints["num_step"]
        self.mpn.load_state_dict(checkpoints["mpn"])
        self.rin.load_state_dict(checkpoints["rin"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizer_mpn.load_state_dict(checkpoints["optimizer_mpn"])
        self.optimizer_rin.load_state_dict(checkpoints["optimizer_rin"])
        self.optimizer_discriminator.load_state_dict(checkpoints["optimizer_D"])
        self.optimizer_joint.load_state_dict(checkpoints["optimizer_joint"])
        self.optimizers_to_cuda()

        # self.scheduler_mpn.load_state_dict(checkpoints["scheduler_mpn"])
        # self.scheduler_rin.load_state_dict(checkpoints["scheduler_rin"])
        # self.scheduler_discriminator.load_state_dict(checkpoints["scheduler_D"])
        # self.scheduler_joint.load_state_dict(checkpoints["scheduler_joint"])

    def optimizers_to_cuda(self):
        for state in self.optimizer_mpn.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_rin.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_joint.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()


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
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"ground_truth",
                                    volume=gt)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"mask",
                                    volume=mask)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"input",
                                    volume=input)
                        tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{global_iter+1}iter",
                                    fileName=f"output",
                                    volume=output)
                    
                global_iter+=1
                

if __name__ == '__main__':
    trainer=UnetTrainer(cfg)
    loader = trainer.data_loader
    imgs, mask= next(iter(loader))
    print(imgs.shape)
    print(mask.shape)
                
                
                