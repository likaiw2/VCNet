from yacs.config import CfgNode as CN

_C = CN()

_C.WANDB = CN()
_C.WANDB.WORK = False
_C.WANDB.LOG_DIR = ""
_C.WANDB.STATUS = "online"

# 保存数据集相关的参数
_C.dataset = CN()
_C.dataset.train_data_path = "/root/autodl-tmp/Diode/Datas/VCNet_dataSet/train"
_C.dataset.data_save_path = "/root/autodl-tmp/Diode/Codes/Volume_Impainting/DCGAN_new/out"
_C.dataset.volume_shape = (128,128,128)
_C.dataset.target_shape = (128,128,128)
_C.dataset.mask_type = "train"

# 保存网络相关的参数
_C.net = CN()
_C.net.model_name = "DCGAN_deep_partial"
_C.net.gen_input_channel = 1
_C.net.gen_dp_prob = 0.2
_C.net.disc_input_channel = 2
_C.net.learning_rate = 0.0002             #原模型参数 5e-3(0.005)
_C.net.batch_size = 1
_C.net.lambda_recon = 200
_C.net.dilation_flag = False
_C.net.trilinear = False
_C.net.partial = True

# 保存训练相关的参数
_C.train = CN()
_C.train.save_model = True
_C.train.save_raw = True
_C.train.device = "cuda:0"
_C.train.total_epoch = 1000
_C.train.log_save_iter = 100






def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# provide a way to import the defaults as a global singleton:
cfg = _C  # users can `from config import cfg`