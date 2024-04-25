import math
from yacs.config import CfgNode as CN
import numpy as np

#创建一个配置节点 _C
_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cpu"

_C.MODEL.NAME = "VCnet_3D"

_C.RUN = CN()
_C.RUN.LOAD_PTH = False
_C.RUN.SAVE_PTH = True
_C.RUN.TYPE = "train"

_C.PATH = CN()
_C.PATH.SOURCE_PATH="/Users/wanglikai/Codes/Volume_Inpainting/dataSet1"
_C.PATH.TEMP_PATH="/Users/wanglikai/Codes/Volume_Inpainting/VCNet_modify/temp_data"
_C.PATH.SAVE_PATH="/Users/wanglikai/Codes/Volume_Inpainting/VCNet_modify/out"
_C.PATH.PTH_LOAD=""
_C.PATH.PTH_SAVE=""

# 在_C下创建新的配置节点_C.SYSTEM
_C.DATASET = CN()
_C.DATASET.DATA_SHAPE = (128,128,128)     #[depth, height, width]
_C.DATASET.DATA_TYPE = np.float32
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
_C.TRAIN.START_STEP = 0
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.TEST_MODE=True
_C.TRAIN.UP_MODE=3

_C.TRAIN.INTERVAL_TOTAL = 2000
_C.TRAIN.INTERVAL_JOINT = 1000
_C.TRAIN.INTERVAL_LOG = 200
_C.TRAIN.INTERVAL_SAVE = 500
_C.TRAIN.INTERVAL_VISUALIZE = 200

_C.TRAIN.LAMBDA_RECON=200
_C.TRAIN.LAMBDA_ADV=1e-3
_C.TRAIN.LAMBDA_REC=1

_C.TRAIN.LEARN_RATE=5e-4
_C.TRAIN.WEIGHT_DECAY_ADV=1e-4
_C.TRAIN.WEIGHT_DECAY_REC=1e-4



_C.TRAIN.MASKED_PIC_PATH=""



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# provide a way to import the defaults as a global singleton:
cfg = _C  # users can `from config import cfg`
