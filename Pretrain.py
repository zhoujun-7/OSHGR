import argparse
from lib.config.default_args import update_args
from lib.engine.ddp_trainer import DDP_Trainer


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-task", type=str, default="pretrain", choices=["pretrain", "openset_HGR", "FSCI", "FSSI"])
    # Training
    parser.add_argument("-epoch", type=int, default=100)
    parser.add_argument("-per_gpu_batch", type=int, default=64)
    parser.add_argument("-num_work", type=int, default=8)
    parser.add_argument("-episode", type=int, default=1000)
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-checkpoint", type=str, default="")
    parser.add_argument("-grad_accumulate", type=int, default=1)
    parser.add_argument("-print_freq", type=int, default=100)
    parser.add_argument("-data_init", action="store_false")
    parser.add_argument("-validation", action="store_false")
    parser.add_argument("-n_shot", type=int, default=5)
    parser.add_argument("-save_each", action="store_true")

    # Optimizer
    parser.add_argument("-optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("-learning_rate", type=float, default=0.0003)
    parser.add_argument("-LR_factor", type=float, default=0.2)
    parser.add_argument("-LR_step", nargs="+", type=int, default=[60, 80])
    parser.add_argument("-WD", type=float, default=0.0005)

    # Loss
    parser.add_argument("-loss_CLS", type=float, default=0)
    parser.add_argument("-loss_MAE", type=float, default=1)
    parser.add_argument("-loss_UVD", type=float, default=0.0005)
    parser.add_argument("-loss_KPT", type=float, default=0.0005)
    parser.add_argument("-loss_ANG", type=float, default=0.0005)
    parser.add_argument("-loss_SYN", type=float, default=0.00015)

    # DDP
    parser.add_argument("-ddp_master_port", type=int, default=29501)
    parser.add_argument("-num_gpu", type=int, default=2)

    # Model
    ## backbone
    # parser.add_argument("-input_noise", action="store_True")
    parser.add_argument("-only_MAE", action="store_true")
    parser.add_argument("-backbone", type=str, default="vit", choices=["resnet18", "vit"])
    ## MAE
    parser.add_argument("-use_MAE", action="store_false")
    parser.add_argument("-MAE_from_noise", action="store_false")
    ## A2J
    parser.add_argument("-use_A2J", action="store_false")
    parser.add_argument("-downsample", type=str, default="spatial", choices=["avg_pool", "spatial"])
    parser.add_argument("-A2J_loc", type=int, default=7)
    parser.add_argument("-detach_spatial_downsample", action="store_false")
    ## GCN
    parser.add_argument("-use_GCN", action="store_false")

    # Dataset
    parser.add_argument("-num_class", type=int, default=68)
    parser.add_argument("-HPE_ratio", type=float, default=1)
    parser.add_argument("-use_sampler", action="store_false")

    return parser


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    args = update_args(args)

    trainer = DDP_Trainer(args)
    # trainer.run()

    trainer.ddp_run(
        trainer.ddp_worker,
        trainer.WORLD_SIZE,
    )
