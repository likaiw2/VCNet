import sys
print(sys.version)

import glog as log  # Google的日志库
import argparse  # 用于解析命令行参数
from trainer import Trainer
# from tester import Tester  # 导入测试器类
from options.config import get_cfg_defaults  # 导入获取默认配置的函数

# 获取默认配置
cfg = get_cfg_defaults()
cfg.merge_from_file("options/macbook.yaml")
# cfg.freeze()

# print(cfg)  # 打印配置信息


if cfg.RUN.TYPE=="train":
    # 创建训练器对象并开始训练
    trainer = Trainer(cfg)
    trainer.run()