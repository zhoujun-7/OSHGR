import torch
from addict import Dict


args = Dict()

args.TRAIN.TASK = "PRETRAIN"
args.TRAIN.EPISODE = 1000
args.TRAIN.CHECKPOINT = ""
args.TRAIN.PER_GPU_BATCH = 64
args.TRAIN.NUM_WORK = 8
args.TRAIN.DTYPE = torch.float32
args.TRAIN.EPOCH = 200
args.TRAIN.SEED = 1
args.TRAIN.DATA_INIT = True
args.TRAIN.VALIDATION = True

args.OPTIM.OPTIMIZER = "adamw"
args.OPTIM.LR = 0.0003
args.OPTIM.LR_FACTOR = 0.2
args.OPTIM.LR_STEP = (80, 120)
args.OPTIM.WD = 0.0005
args.OPTIM.MOMENTUM = 0.9
args.OPTIM.GRAD_ACCM = 8

args.LOG.DIR = "log"
args.LOG.PRINT_FREQ = 100

args.LOSS.CLS = 0
args.LOSS.MAE = 1
args.LOSS.UVD = 0.0005
args.LOSS.KPT = 0.0005
args.LOSS.ANG = 0.0005
args.LOSS.SYN = 0.00015

args.DDP.MASTER_PORT = 29500
args.DDP.WORLD_SIZE = 2

args.MODEL.BACKBONE = "vit"
# args.MODEL.VIT.TYPE = "vit_base"
# args.MODEL.VIT.POOL = "token"  # ['token', 'global_pool', 'spatial']
args.MODEL.MAE.USE = True
args.MODEL.MAE.FROM_NOISE = True
args.MODEL.A2J.USE = True
args.MODEL.A2J.LOC = 7
args.MODEL.A2J.DETACH = False
args.MODEL.DOWNSAMPLE = "spatial"  # ['avg_pool']
args.MODEL.GCN.USE = True
args.MODEL.GCN.IN_DIM = 768
args.MODEL.GCN.HID_DIM = 128

args.DATASET.NUM_CLASS = 68
args.DATASET.NUM_BASE_CLASS = 23
args.DATASET.NUM_INCR_CLASS = args.DATASET.NUM_CLASS - args.DATASET.NUM_BASE_CLASS
args.DATASET.RESOLUTION = (224, 224)
args.DATASET.DEPTH_NORMALIZE = 150
args.DATASET.INCR_WAY = 5
args.DATASET.INCR_SHOT = 5
args.DATASET.NUM_SESSION = args.DATASET.NUM_INCR_CLASS // args.DATASET.INCR_WAY
args.DATASET.PRETRAIN_SHOT = 4
args.DATASET.BIGHAND_DIR = "data/BigHand/bighand_crop"
args.DATASET.OSHGR_DIR = "data/HGR68/OSHGR"
args.DATASET.HGR_HPE_RATIO = [1, 1]

args.AUGMENT.SHIFT_U = (-30, 30)
args.AUGMENT.SHIFT_V = (-30, 30)
args.AUGMENT.SHIFT_D = (-30, 30)
args.AUGMENT.ROTATION = (-180, 180)
args.AUGMENT.SCALE = (0.8, 1.3)
args.AUGMENT.GAUSS_NOISE_PROBABILITY = 0.5
args.AUGMENT.GAUSS_NOISE_MU = (-3, 3)
args.AUGMENT.GAUSS_NOISE_SIGMA = (3, 30)
args.AUGMENT.ERASER_PROBABILITY = 0.5
args.AUGMENT.ERASE_RATIO = (0.02, 0.4)
args.AUGMENT.ERASE_PATH_RATIO = 0.3
args.AUGMENT.ERASE_MU = (-3, 3)
args.AUGMENT.ERASE_SIGMA = (30, 80)
args.AUGMENT.SMOOTH_PROBABILITY = 0.5
args.AUGMENT.SMOOTH_KERNEL = (2, 5)
args.AUGMENT.IS_AUGMENT = True
args.AUGMENT.INCR_AUGMENT = False
args.AUGMENT.NUM_INCR_AUGMENT = 1


def update_args(cmd_args):
    if cmd_args.task == "pretrain":
        args.TRAIN.EPOCH = cmd_args.epoch
        args.TRAIN.EPISODE = cmd_args.episode
        args.TRAIN.DATA_INIT = cmd_args.data_init
        args.TRAIN.VALIDATION = cmd_args.validation
        args.TRAIN.SAVE_EACH = cmd_args.save_each

        args.DATASET.HGR_HPE_RATIO[1] = cmd_args.HPE_ratio

        args.OPTIM.OPTIMIZER = cmd_args.optimizer
        args.OPTIM.GRAD_ACCM = cmd_args.grad_accumulate
        args.OPTIM.LR = cmd_args.learning_rate
        args.OPTIM.LR_FACTOR = cmd_args.LR_factor
        args.OPTIM.LR_STEP = cmd_args.LR_step
        args.OPTIM.WD = cmd_args.WD

        args.LOG.PRINT_FREQ = cmd_args.print_freq

        args.LOSS.CLS = cmd_args.loss_CLS
        args.LOSS.MAE = cmd_args.loss_MAE
        args.LOSS.UVD = cmd_args.loss_UVD
        args.LOSS.KPT = cmd_args.loss_KPT
        args.LOSS.ANG = cmd_args.loss_ANG
        args.LOSS.SYN = cmd_args.loss_SYN

        args.DDP.MASTER_PORT = cmd_args.ddp_master_port
        args.DDP.WORLD_SIZE = cmd_args.num_gpu
        args.MODEL.ONLY_MAE = cmd_args.only_MAE
        args.DATASET.USE_SAMPLER = cmd_args.use_sampler

    args.TRAIN.TASK = cmd_args.task
    args.TRAIN.PER_GPU_BATCH = cmd_args.per_gpu_batch
    args.TRAIN.NUM_WORK = cmd_args.num_work
    args.TRAIN.SEED = cmd_args.seed
    args.TRAIN.CHECKPOINT = cmd_args.checkpoint

    args.MODEL.BACKBONE = cmd_args.backbone
    args.MODEL.MAE.USE = cmd_args.use_MAE
    args.MODEL.MAE.FROM_NOISE = cmd_args.MAE_from_noise
    args.MODEL.A2J.USE = cmd_args.use_A2J
    args.MODEL.A2J.LOC = cmd_args.A2J_loc
    args.MODEL.A2J.DETACH = cmd_args.detach_spatial_downsample
    args.MODEL.DOWNSAMPLE = cmd_args.downsample
    args.MODEL.GCN.USE = cmd_args.use_GCN

    args.DATASET.INCR_SHOT = cmd_args.n_shot

    if args.MODEL.BACKBONE == "resnet18":
        args.MODEL.MAE.USE = False
        args.MODEL.A2J.USE = False
        args.MODEL.GCN.USE = False

    if not args.MODEL.A2J.USE:
        args.MODEL.DOWNSAMPLE = "avg_pool"

    try:
        args.AUGMENT_PROTO.N_BASE_DATA = cmd_args.n_augment_basedata
        args.AUGMENT_PROTO.N_INCR_DATA = cmd_args.n_augment_incrdata
        args.AUGMENT_PROTO.N_PROTO = cmd_args.n_augment_proto
        args.AUGMENT_PROTO.N_R_PROTO = cmd_args.n_augment_r_proto
    except:
        pass

    return args
