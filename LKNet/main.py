import glog as log  # Google的日志库
import trainer
import sys
# from tester import Tester  # 导入测试器类
from configs.config import get_cfg_defaults  # 导入获取默认配置的函数
import model.models as models

# 获取默认配置
cfg = get_cfg_defaults()

plat = sys.platform
if plat=="darwin":                                  # MacOS
    print("MacOS")
    cfg.merge_from_file("configs/macbook.yaml")
elif plat=="linux":                                 # linux server
    print("linux server")
    cfg.merge_from_file("configs/linuxserver.yaml")
elif (plat=="win32" or plat=="cygwin"):             # windows
    print("windows")
    cfg.merge_from_file(r"Volume_Inpainting\LKNet\configs\windows.yaml")
else:
    print("can't judge platform automatically,please check yaml path")
    # cfg.merge_from_file(r"Volume_Inpainting\VCNet_modify\configs\windows.yaml")
    cfg.merge_from_file(None)


# cfg.freeze()

# print(cfg)  # 打印配置信息

Unet_model = {
    "PconvUnet": models.PConvUNet,
}

GAN_model = {
    "SAGAN": {"GEN" : models.InpaintSANet,
              "DIS" : models.InpaintSADirciminator},
}


if cfg.RUN.TYPE=="train":
    # 创建训练器对象并开始训练
    # 判断用GAN还是Unet来训练
    if cfg.RUN.MODEL in Unet_model:
        print("Unet!")
        trainer = trainer.UnetTrainer(cfg,
                                      model=Unet_model[cfg.RUN.MODEL]())
        trainer.run()
    elif cfg.RUN.MODEL in GAN_model:
        print("GAN!")
        trainer = trainer.GAN_Trainer(cfg,
                                      net_G=GAN_model[cfg.RUN.MODEL]["GEN"](), 
                                      net_D=GAN_model[cfg.RUN.MODEL]["DIS"]())
        trainer.run()
    else:
        assert True,"Check your model name!"
    