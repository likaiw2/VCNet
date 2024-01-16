from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from torchvision import transforms
import time
import datetime
from torch.autograd import Variable
import tools
import numpy as np

# 设置路径
dataSourcePath = "dataSet1/brain/"
dataSavePath = "output/trial08/brain/"
device=torch.device("cuda:0")
# device=torch.device("cpu")

# 使用前清除cuda缓存
torch.cuda.empty_cache()
torch.manual_seed(0)

# 对3d卷积神经网络的权重初始化
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


#模型的代码实现见VCNet_model.py
from VCNet.VCNet_model import *

# Feel free to change pretrained to False if you're training the model from scratch
pretrained = False
save_model = False

fileStartVal = 1
fileIncrement = 1
constVal = 1



#---------------------initialize the model----------------------
# 1)parameters for dataset
total_index = 100
ratio = 0.7
maxi_train_index = total_index*ratio
dim = (160, 224, 168)   # [depth, height, width]. brain
float32DataType = np.float32

trainDataset = tools.DataSet(data_path="",
                             mask_type="test",
                             prefix="norm_ct",
                             data_type="raw",
                             volume_shape=dim,
                             max_index=70)

# 2)parameters for loss function
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()

# 3)other parameters
lambda_recon = 200
n_epochs = 1000
input_dim = 1
real_dim = 1
batch_size = 1          #原模型参数 10
lr = 5e-3             #learn rate 原模型参数 5e-3(0.005)


# 4)send parameters to cuda
gen = UNet_v2(in_channel=1).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Dis_VCNet().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-3)

criterion_bce = nn.BCELoss().to(device)
criterion_L1 = nn.L1Loss().to(device)
criterion_L2 = nn.MSELoss().to(device)


if pretrained:
    loaded_state = torch.load("output/trial8/DCGAN_27999.pth")
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    disc.load_state_dict(loaded_state["disc"])
    disc_opt.load_state_dict(loaded_state["disc_opt"])
else:
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

# print("Success!")

#---------------------------------training------------------------
def train(save_model=True):
    # read the start time
    ot = time.time()
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("##train start(brain)##  time:",t1)
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    running_loss = 0.0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for ct,mri,index in dataloader:

            # wrap them into Variable
            ct = ct.to(device)
            mri = mri.to(device)
            # print("ct: " , ct.shape)
            # print("mri: " , mri.shape)

            ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # source->ori_ct  residual_source->ori_ct  label->ori_mri  outputG->fake
            # real_label = torch.ones(batch_size,1)  fake_label = torch.zeros(batch_size,1)
            # 初始化变量
            disc_opt.zero_grad()  # Zero out the gradient before backpropagation
            fake = gen(ct)

            source = Variable(ct)
            labels = Variable(mri)
            outputG = Variable(fake)

            outputD_real = disc(labels)
            outputD_real = F.sigmoid(outputD_real)

            outputD_fake = disc(outputG).detach()
            outputD_fake = F.sigmoid(outputD_fake)
            disc.zero_grad()
            real_label = torch.ones(batch_size, 1)
            real_label = real_label.to(device)
            # print(real_label.size())
            real_label = Variable(real_label)
            # print(outputD_real.size())
            loss_real = criterion_bce(outputD_real, real_label)
            loss_real.backward()
            # train with fake data
            fake_label = torch.zeros(batch_size, 1)
            #         fake_label = torch.FloatTensor(batch_size)
            #         fake_label.data.resize_(batch_size).fill_(0)
            fake_label = fake_label.to(device)
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake, fake_label).requires_grad_(True)
            loss_fake.backward()

            # lossD = loss_real + loss_fake
            lossD = loss_real + loss_fake
            # update network parameters
            disc_opt.step()

            ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))------------------------

            #         print inputs.data.shape
            # outputG = net(source) #here I am not sure whether we should use twice or not
            outputG = gen(source)  # 5x64x64->1*64x64

            # outputG = net(source,residual_source) #5x64x64->1*64x64
            gen.zero_grad()
            lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))
            lossG_G = 1 * lossG_G
            lossG_G.backward()  # compute gradients

            outputG = gen(source)  # 5x64x64->1*64x64

            if len(outputG.size()) == 3:
                outputG = outputG.unsqueeze(1)

            outputD = disc(outputG)
            outputD = F.sigmoid(outputD)
            lossG_D = 0.05 * criterion_bce(outputD,real_label)  # note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward()
            # for other losses, we can define the loss function following the pytorch tutorial
            gen_opt.step()  # update network parameters
            running_loss = running_loss + lossG_G
            #----------------------------------------------------------------------------------

            ### Visualization code ###
            if (cur_step+1) % display_step == 0 or cur_step == 1:
                
                t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                print(t, f"  Epoch {epoch}: Step {cur_step}: Generator (Res_UNet) loss: {running_loss/100}, Discriminator loss: {lossD}")
                #计算单位运行时间
                dt = time.time() - ot
                elapsedTime = str(datetime.timedelta(seconds=dt))
                per_epoch = str(datetime.timedelta(seconds=dt / (epoch+1)))
                print(f"    epoch = {epoch}     dt={elapsedTime}    per-epoch={per_epoch}")
                # save fake.
                saveRawFile10(cur_step,
                              'fake_mr',
                              (fileStartVal + index * fileIncrement) / constVal,
                              '',
                              fake[0, 0, :, :, :])

                saveRawFile10(cur_step, 'truth_mr', (fileStartVal + index * fileIncrement) / constVal, '',
                              mri[0, 0, :, :, :])
                
                # show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                # show_tensor_images(real, size=(real_dim, target_shape, target_shape))
                # show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    saveModel(cur_step=cur_step)
                    
            cur_step += 1
    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("##train finished(brain)##  time:",t2)
    print("total train time:")
    print("start:",t1)
    print("end:",t2)
train()