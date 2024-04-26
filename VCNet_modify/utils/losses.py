import numpy as np
import torch
import torch.nn as nn
import utils.tools as tools

# LOSS Function Class
class WeightedMSELoss(nn.Module):
    # $\mathcal{L}_{\mathrm{rec}}^G=\frac{1}{n} \sum_{j=1}^n\left\|\mathbf{M}_j^C \odot\left(G\left(\mathbf{V}_{M, j}^C\right)-\mathbf{V}_j^C\right)\right\|_2$
    
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, ground_truth, net_output,mask):
        batch_size = ground_truth.shape[0]
        # print(mask.shape)
        iter_norm = []
        
        for i in range(batch_size):
            V_ground_truth = ground_truth[i][0]
            V_net_output = net_output[i][0]
            V_mask = mask[i][0]
            diff = V_net_output - V_ground_truth
            valuable_part = V_mask * diff
            # valuable_part_norm = np.linalg.norm(valuable_part,ord=2)
            valuable_part_norm = torch.linalg.norm(valuable_part,dim=1,ord=2).cpu().detach().numpy()
            iter_norm.append(valuable_part_norm)
            
        iter_norm = np.array(iter_norm)
        
        loss_WeightedMSE = torch.mean(torch.tensor(iter_norm,requires_grad=True).cuda())

        return loss_WeightedMSE
    
class AdversarialGLoss(nn.Module):
    # $\mathcal{L}_{\mathrm{adv}}^G=\frac{1}{n} \sum_{j=1}^n\left[\log D\left(\mathbf{M}_j^C \odot G\left(\mathbf{V}_{M, j}^C\right)+\left(\mathbf{1}-\mathbf{M}_j^C\right) \odot \mathbf{V}_j^C\right)\right]$
    def __init__(self, discriminator):
        super(AdversarialGLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, ground_truth, net_output,mask):
        batch_size = ground_truth.shape[0]
        # print(mask.shape)
        iter_norm = []
        
        for i in range(batch_size):
            # V_ground_truth = ground_truth[i][0]
            # V_net_output = net_output[i][0]
            # V_mask = mask[i]
            
            mixed_data = mask * net_output + (1 - mask) * ground_truth
            dis_output = self.discriminator(mixed_data)
            
            log_dis = torch.log10(dis_output).cpu().detach().numpy()
            
            iter_norm.append(log_dis)
            
        iter_norm = np.array(iter_norm)
        
        adversarial_loss = torch.mean(torch.tensor(iter_norm,requires_grad=True).cuda())

        return adversarial_loss

class AdversarialDLoss(nn.Module):
    def __init__(self, discriminator):
        super(AdversarialDLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, ground_truth, net_output,mask):
        v_mask=mask
        batch_size = ground_truth.shape[0]
        # print(mask.shape)
        exp1 = []
        exp2 = []
        
        for i in range(batch_size):

            # print("net_output:",net_output.shape)
            log_dis1 = torch.log10(self.discriminator(ground_truth)).cpu().detach().numpy()
            exp1.append(log_dis1)
            
            mixed_data = mask * net_output + (1 - mask) * ground_truth
            # print("mixed_data:",mixed_data.shape)
            dis_output = self.discriminator(mixed_data)
            
            log_dis2 = torch.log10(1 - dis_output).cpu().detach().numpy()
            exp2.append(log_dis2)
            
        exp1 = np.array(exp1)
        exp2 = np.array(exp2)
        
        adversarial_loss1 = torch.mean(torch.tensor(exp1,requires_grad=True).cuda())
        adversarial_loss2 = torch.mean(torch.tensor(exp2,requires_grad=True).cuda())
        adversarial_loss = adversarial_loss1 + adversarial_loss2

        return adversarial_loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor=tools.VGG16FeatureExtractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(self.gram_matrix(feat_output[i]),
                                          self.gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(self.gram_matrix(feat_output_comp[i]),
                                          self.gram_matrix(feat_gt[i]))

        loss_dict['tv'] = self.total_variation_loss(output_comp)

        return loss_dict
    
    def gram_matrix(self,feat):
        # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram
    
    def total_variation_loss(self,image):
        # shift one pixel and get difference (for both x and y direction)
        loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
            torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return loss


