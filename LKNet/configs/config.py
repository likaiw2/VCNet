import math
from yacs.config import CfgNode as CN
import numpy as np

#创建一个配置节点 _C
_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cpu"
_C.SYSTEM.NUM_WORKERS = 0

_C.RUN = CN()
_C.RUN.LOAD_PTH = False
_C.RUN.SAVE_PTH = True
_C.RUN.TYPE = "train"
_C.RUN.MODEL = "SAGAN" #"PconvUnet"
_C.RUN.ADD_INFO = "5.4"

_C.PATH = CN()
_C.PATH.DATA_PATH="Please set path in yaml"
_C.PATH.CONT_PATH="Please set path in yaml"
_C.PATH.TEMP_PATH="Please set path in yaml"
_C.PATH.SAVE_PATH="Please set path in yaml"
_C.PATH.PTH_LOAD_PATH="Please set path in yaml"
_C.PATH.PTH_SAVE_PATH="Please set path in yaml"
_C.PATH.VGG16_PATH="Please set path in yaml"


_C.DATASET = CN()
# _C.DATASET.ORIGIN_SHAPE = (160,224,168)     #[depth, height, width]
_C.DATASET.ORIGIN_SHAPE = (128,128,128)     #[depth, height, width]
_C.DATASET.TARGET_SHAPE = (128,128,128)     #[depth, height, width]
_C.DATASET.DATA_TYPE = "np.float32"
_C.DATASET.MEAN = [0.5, 0.5, 0.5]
_C.DATASET.STD = [0.5, 0.5, 0.5]
_C.DATASET.SHUFFLE = True

_C.WANDB = CN()
_C.WANDB.PROJECT_NAME = "vcnet"
_C.WANDB.RUN = 1
_C.WANDB.LOG_DIR = ""
_C.WANDB.NUM_ROW = 0
_C.WANDB.MODE = "offline"

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.TEST_MODE=True
_C.TRAIN.UP_MODE=3

_C.TRAIN.INTERVAL_START = 0
_C.TRAIN.INTERVAL_TOTAL = 100000
_C.TRAIN.INTERVAL_JOINT = 1000
_C.TRAIN.INTERVAL_LOG = 200
_C.TRAIN.INTERVAL_SAVE = 4000
_C.TRAIN.INTERVAL_VISUALIZE = 4000
_C.TRAIN.EPOCH_TOTAL = 1000

_C.TRAIN.LAMBDA_RECON=200
_C.TRAIN.LAMBDA_ADV=1e-3
_C.TRAIN.LAMBDA_REC=1

_C.TRAIN.LEARN_RATE=5e-4
_C.TRAIN.WEIGHT_DECAY_ADV=1e-4
_C.TRAIN.WEIGHT_DECAY_REC=1e-4

_C.MODEL = CN()
_C.MODEL.GEN = CN()
_C.MODEL.DIS = CN()



_C.TRAIN.MASKED_PIC_PATH=""



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# provide a way to import the defaults as a global singleton:
cfg = _C  # users can `from config import cfg`
