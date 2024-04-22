import os
import torch
import wandb
import glog as log
import numpy as np

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from modeling.architecture import MPN, RIN, Discriminator, PatchDiscriminator
from losses.bce import WeightedBCELoss
from losses.consistency import SemanticConsistencyLoss, IDMRFLoss
from losses.adversarial import compute_gradient_penalty
from utils.mask_utils import MaskGenerator, ConfidenceDrivenMaskLayer, COLORS
from utils.data_utils import linear_scaling, linear_unscaling, get_random_string, RaindropDataset, NormalDataset

# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, cfg):
        self.opt = cfg
        # 根据配置参数构造模型名称
        self.model_name = f"{self.opt.MODEL.NAME}_{self.opt.DATASET.NAME}_{self.opt.TRAIN.NUM_TOTAL_STEP}step_{self.opt.TRAIN.BATCH_SIZE}bs_{self.opt.MODEL.JOINT.LR}lr"
                          
        # 设置wandb的日志目录
        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        cfg.freeze()
        
        # 初始化wandb，用于实验跟踪和可视化
        self.wandb = wandb
        # self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, notes=self.opt.WANDB.LOG_DIR, config=self.opt, entity=self.opt.WANDB.ENTITY)
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, notes=self.opt.WANDB.LOG_DIR, config=self.opt)
        
        # 定义图像预处理流程
        self.transform = transforms.Compose([transforms.Resize(self.opt.DATASET.SIZE),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                             ])
        # 创建一个图像数据集
        self.dataset = ImageFolder(root=self.opt.DATASET.PATH, transform=self.transform)
        # 创建一个图像数据加载器
        self.image_loader = data.DataLoader(dataset=self.dataset, 
                                            batch_size=self.opt.TRAIN.BATCH_SIZE, 
                                            shuffle=self.opt.TRAIN.SHUFFLE, 
                                            num_workers=self.opt.SYSTEM.NUM_WORKERS)
        
        # 定义另一个图像预处理流程，用于content images(用来填充mask区域的图像)
        self.imagenet_transform = transforms.Compose([transforms.RandomCrop(self.opt.DATASET.SIZE, pad_if_needed=True, padding_mode="reflect"),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                                      ])
        # # 根据数据集名称创建不同的content datasets
        # if self.opt.DATASET.NAME.lower() == "ffhq":
        #     celeb_dataset = ImageFolder(root=self.opt.DATASET.CONT_ROOT, transform=self.transform)
        #     imagenet_dataset = ImageFolder(root=self.opt.DATASET.IMAGENET, transform=self.imagenet_transform)
        #     self.cont_dataset = torch.utils.data.ConcatDataset([celeb_dataset, imagenet_dataset])
        # else:
        #     # 创建单一数据集
        #     self.cont_dataset = ImageFolder(root=self.opt.DATASET.CONT_ROOT, transform=self.imagenet_transform)
        # self.cont_dataset = ImageFolder(root=self.opt.DATASET.CONT_ROOT, transform=self.imagenet_transform)
        self.cont_dataset = NormalDataset(root=self.opt.DATASET.CONT_ROOT, transform=self.imagenet_transform)
        
        # 创建content images的数据加载器 
        self.cont_image_loader = data.DataLoader(dataset=self.cont_dataset, 
                                                 batch_size=self.opt.TRAIN.BATCH_SIZE, 
                                                 shuffle=self.opt.TRAIN.SHUFFLE, 
                                                 num_workers=self.opt.SYSTEM.NUM_WORKERS)
        
        # 创建遮罩生成器和平滑器
        self.mask_generator = MaskGenerator(self.opt.MASK)
        self.mask_smoother = ConfidenceDrivenMaskLayer(self.opt.MASK.GAUS_K_SIZE, self.opt.MASK.SIGMA)
        # self.mask_smoother = GaussianSmoothing(1, 5, 1/40)
        
        # 定义将张量转换为PIL图像的函数
        self.to_pil = transforms.ToPILImage()

        # 创建模型组件
        self.mpn = MPN(base_n_channels=self.opt.MODEL.MPN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.rin = RIN(base_n_channels=self.opt.MODEL.RIN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
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
            cont_imgs, _ = next(iter(self.cont_image_loader))
            cont_imgs = linear_scaling(cont_imgs.float().cuda())
            if cont_imgs.size(0) != imgs.size(0):
                cont_imgs = cont_imgs[:imgs.size(0)]

            # 应用平滑遮罩
            smooth_masks = self.mask_smoother(1 - masks) + masks
            smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

            # 根据遮罩合成图像
            masked_imgs = cont_imgs * smooth_masks + imgs * (1. - smooth_masks)
            
            self.unknown_pixel_ratio = torch.sum(masks.view(batch_size, -1), dim=1).mean() / (h * w)

            # 训练判别器
            for _ in range(self.opt.MODEL.D.NUM_CRITICS):
                d_loss = self.train_D(masked_imgs, masks, y_imgs)
            info += "D Loss: {} ".format(d_loss)

            # 训练生成器
            m_loss, g_loss, pred_masks, output = self.train_G(masked_imgs, masks, y_imgs)
            info += "M Loss: {} G Loss: {} ".format(m_loss, g_loss)

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
            if self.num_step % self.opt.TRAIN.SAVE_INTERVAL == 0 and self.num_step != 0:
                self.do_checkpoint(self.num_step)

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

    def do_checkpoint(self, num_step):
        # 创建保存检查点的目录
        if not os.path.exists("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name)):
            os.makedirs("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name), exist_ok=True)

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
        torch.save(checkpoint, "./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))

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


class RaindropTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        assert self.opt.DATASET.NAME.lower() == "raindrop"

        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.JOINT.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)

        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, notes=self.opt.WANDB.LOG_DIR, config=self.opt, entity=self.opt.WANDB.ENTITY)

        self.transform = transforms.Compose([transforms.Resize(self.opt.DATASET.SIZE),
                                             transforms.CenterCrop(self.opt.DATASET.SIZE),
                                             transforms.ToTensor()
                                             ])
        self.dataset = RaindropDataset(root=self.opt.DATASET.RAINDROP_ROOT, transform=self.transform, target_transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)

        self.to_pil = transforms.ToPILImage()

        self.mpn = MPN(base_n_channels=self.opt.MODEL.MPN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.rin = RIN(base_n_channels=self.opt.MODEL.RIN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.discriminator = Discriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)

        self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()), lr=self.opt.MODEL.D.LR, betas=self.opt.MODEL.D.BETAS)
        self.optimizer_joint = torch.optim.Adam(list(self.mpn.parameters()) + list(self.rin.parameters()), lr=self.opt.MODEL.JOINT.LR, betas=self.opt.MODEL.JOINT.BETAS)

        self.num_step = self.opt.TRAIN.START_STEP

        log.info("Checkpoints loading...")
        self.load_checkpoints(self.opt.TRAIN.START_STEP)

        self.check_and_use_multi_gpu()

        self.reconstruction_loss = torch.nn.L1Loss().cuda()
        self.semantic_consistency_loss = SemanticConsistencyLoss().cuda()
        self.texture_consistency_loss = IDMRFLoss().cuda()

    def run(self):
        print("start run!")
        while self.num_step < self.opt.TRAIN.NUM_TOTAL_STEP:
            self.num_step += 1
            info = " [Step: {}/{} ({}%)] ".format(self.num_step, self.opt.TRAIN.NUM_TOTAL_STEP, 100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP)

            imgs, y_imgs = next(iter(self.image_loader))
            imgs = linear_scaling(imgs.float().cuda())
            y_imgs = y_imgs.float().cuda()

            for _ in range(self.opt.MODEL.D.NUM_CRITICS):
                self.optimizer_discriminator.zero_grad()

                pred_masks, neck = self.mpn(imgs)
                output = self.rin(imgs, pred_masks, neck)

                real_validity = self.discriminator(y_imgs).mean()
                fake_validity = self.discriminator(output.detach()).mean()
                gp = compute_gradient_penalty(self.discriminator, output.data, y_imgs.data)

                d_loss = -real_validity + fake_validity + self.opt.OPTIM.GP * gp
                d_loss.backward()
                self.optimizer_discriminator.step()

                self.wandb.log({"real_validity": -real_validity.item(),
                                "fake_validity": fake_validity.item(),
                                "gp": gp.item()}, commit=False)

            self.optimizer_joint.zero_grad()
            pred_masks, neck = self.mpn(imgs)
            if self.opt.MODEL.RIN.EMBRACE:
                x_embraced = imgs.detach() * (1 - pred_masks.detach())
                output = self.rin(x_embraced, pred_masks.detach(), neck.detach())
            else:
                output = self.rin(imgs, pred_masks.detach(), neck.detach())
            recon_loss = self.reconstruction_loss(output, y_imgs)
            sem_const_loss = self.semantic_consistency_loss(output, y_imgs)
            tex_const_loss = self.texture_consistency_loss(output, y_imgs)
            adv_loss = -self.discriminator(output).mean()

            g_loss = self.opt.OPTIM.RECON * recon_loss + \
                     self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                     self.opt.OPTIM.TEXTURE * tex_const_loss + \
                     self.opt.OPTIM.ADVERSARIAL * adv_loss

            g_loss.backward()
            self.optimizer_joint.step()
            self.wandb.log({"recon_loss": recon_loss.item(),
                            "sem_const_loss": sem_const_loss.item(),
                            "tex_const_loss": tex_const_loss.item(),
                            "adv_loss": adv_loss.item()}, commit=False)

            info += "D Loss: {} ".format(d_loss)
            info += "G Loss: {} ".format(g_loss)

            if self.num_step % self.opt.MODEL.RAINDROP_LOG_INTERVAL == 0:
                log.info(info)

            if self.num_step % self.opt.MODEL.RAINDROP_VISUALIZE_INTERVAL == 0:
                idx = self.opt.WANDB.NUM_ROW
                self.wandb.log({"examples": [
                    self.wandb.Image(self.to_pil(y_imgs[idx].cpu()), caption="original_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(imgs[idx]).cpu()), caption="masked_image"),
                    self.wandb.Image(self.to_pil(pred_masks[idx].cpu()), caption="predicted_masks"),
                    self.wandb.Image(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()), caption="output")
                ]}, commit=False)
            self.wandb.log({})
            if self.num_step % self.opt.MODEL.RAINDROP_SAVE_INTERVAL == 0 and self.num_step != 0:
                self.do_checkpoint(self.num_step)

    def do_checkpoint(self, num_step):
        if not os.path.exists("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name)):
            os.makedirs("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name), exist_ok=True)

        checkpoint = {
            'num_step': num_step,
            'mpn': self.mpn.state_dict(),
            'rin': self.rin.state_dict(),
            'D': self.discriminator.state_dict(),
            'optimizer_joint': self.optimizer_joint.state_dict(),
            'optimizer_D': self.optimizer_discriminator.state_dict(),
            # 'scheduler_mpn': self.scheduler_mpn.state_dict(),
            # 'scheduler_rin': self.scheduler_rin.state_dict(),
            # 'scheduler_joint': self.scheduler_joint.state_dict(),
            # 'scheduler_D': self.scheduler_discriminator.state_dict(),
        }
        torch.save(checkpoint, "./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))

    def load_checkpoints(self, num_step):
        checkpoints = torch.load(self.opt.MODEL.RAINDROP_WEIGHTS)
        self.num_step = checkpoints["num_step"]
        self.mpn.load_state_dict(checkpoints["mpn"])
        self.rin.load_state_dict(checkpoints["rin"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.optimizers_to_cuda()

    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
            log.info("Using {} GPUs...".format(torch.cuda.device_count()))
            self.mpn = torch.nn.DataParallel(self.mpn).cuda()
            self.rin = torch.nn.DataParallel(self.rin).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
        else:
            log.info("GPU ID: {}".format(torch.cuda.current_device()))
            self.mpn = self.mpn.cuda()
            self.rin = self.rin.cuda()
            self.discriminator = self.discriminator.cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_joint.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

