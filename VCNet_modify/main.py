import glog as log  # Google的日志库
import trainer
# from tester import Tester  # 导入测试器类
from configs.config import get_cfg_defaults  # 导入获取默认配置的函数
import model.models as models

# 获取默认配置
cfg = get_cfg_defaults()
cfg.merge_from_file("/Users/wanglikai/Codes/Volume_Inpainting/VCNet_modify/configs/macbook.yaml")
# cfg.freeze()

# print(cfg)  # 打印配置信息

Unet_model = {
    "PconvUnet": models.PConvUNet,
}

GAN_model = {
    
}


if cfg.RUN.TYPE=="train":
    # 创建训练器对象并开始训练
    # 判断用GAN还是Unet来训练
    if cfg.RUN.MODEL in Unet_model:
        print("Unet!")
        trainer = trainer.UnetTrainer(cfg,model=Unet_model[cfg.RUN.MODEL]())
        trainer.run()
    elif cfg.RUN.MODEL in GAN_model:
        print("GAN!")
        trainer = GAN_model[cfg.RUN.MODEL]()
        trainer.run()
    else:
        assert True,"Check your model name!"
    