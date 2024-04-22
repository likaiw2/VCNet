import sys
print(sys.version)

import glog as log  # Google的日志库
import argparse  # 用于解析命令行参数
from engine.trainer import Trainer, RaindropTrainer  # 导入训练器类
# from engine.tester import Tester  # 导入测试器类
from utils.config import get_cfg_defaults  # 导入获取默认配置的函数

# 获取默认配置
cfg = get_cfg_defaults()
cfg.merge_from_file(r"LKNet\utils\configs\exp1.yaml")
# cfg.freeze()

# print(cfg)  # 打印配置信息

if cfg.MODEL.IS_TRAIN:
    # 创建训练器对象并开始训练
    trainer = Trainer(cfg)
    trainer.run()

# # 根据配置决定是训练还是测试
# if cfg.MODEL.IS_TRAIN:
#     # 创建训练器对象并开始训练
#     trainer = Trainer(cfg)
#     trainer.run()
# else:
#     # 创建测试器对象进行测试
#     tester = Tester(cfg)
#     if cfg.TEST.ABLATION:
#         # 进行消融研究
#         for i_id in list(range(250, 500)):
#             for c_i_id in list(range(185, 375)):
#                 for mode in list(range(1, 9)):
#                     tester.do_ablation(mode=mode, img_id=i_id, c_img_id=c_i_id)
#                     log.info("I: {}, C: {}, Mode:{}".format(i_id, c_i_id, mode))
#     else:
#         # 进行定性测试
#         # 以下是一系列的测试用例，包括不同的测试模式和图像路径
        
#         # qualitative
#         img_path = "datasets/ffhq/images1024x1024/07000/07042.png"
#         img_path_2 = "datasets/ffhq/images1024x1024/02000/02056.png"
#         cont_path = "datasets/CelebAMask-HQ/CelebA-HQ-img/1147.jpg"
#         in_cont_path = "datasets/ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG"
#         mask_path = "../../Downloads/mask4.jpg"
#         graf_path = "../../Downloads/graf2.png"
#         graf_mask_path = "../../Downloads/graf-mask-2.jpeg"
#         # tester.infer(img_path, img_path_2, mask_path=mask_path, mode=1, output_dir="../../Downloads")
#         # tester.infer(img_path, in_cont_path, mask_path=mask_path, mode=1, output_dir="../../Downloads")
#         # tester.infer(img_path, mask_path=mask_path, mode=2, output_dir="../../Downloads")
#         # tester.infer(img_path, mode=3, mask_path=mask_path, color="RED", output_dir="../../Downloads")
#         # tester.infer(img_path, mode=3, mask_path=mask_path, color="BLUE", output_dir="../../Downloads")
#         # tester.infer(img_path, mode=3, mask_path=mask_path, color="GREEN", output_dir="../../Downloads")
#         # tester.infer(img_path, mode=3, mask_path=mask_path, color="WHITE", output_dir="../../Downloads")
#         # tester.infer(img_path, mode=3, mask_path=mask_path, color="BLACK", output_dir="../../Downloads")

#         # tester.infer(img_path, mode=4) # ???
#         # tester.infer(img_path, graf_path, mask_path=graf_mask_path, mode=5)
#         # tester.infer(img_path, cont_path, mode=6)  # problematic
#         # tester.infer(img_path, cont_path, mode=7, text="furkan", color="BLUE")  # problematic
#         # tester.infer(img_path, img_path_2, mask_path=mask_path, mode=8, output_dir="../../Downloads")  # problematic

#         raindrop_img_path = "datasets/raindrop/train20/test/data/4_rain.png"
#         raindrop_gt_path = "datasets/raindrop/train20/test/gt/4_clean.png"
#         tester.infer(raindrop_img_path, mode=4, gt_path=raindrop_gt_path)
#         # quantitative
#         # tester.eval()
