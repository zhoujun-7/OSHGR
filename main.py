import argparse
from lib.config.default_args import update_args
from lib.engine.angle_incremental_trainer import IncrementalTrainer


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-task", type=str, default="UFSCI", choices=["UFSCI", "UFSSI", "CFSCI", "CFSSI"])
    # Training
    parser.add_argument("-per_gpu_batch", type=int, default=64)
    parser.add_argument("-num_work", type=int, default=8)
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-checkpoint", type=str, default="log/Train/Ours/best/8.pth")
    parser.add_argument("-n_shot", type=int, default=5)

    # Model
    ## backbone
    # parser.add_argument("-input_noise", action="store_True")
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

    return parser


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    args = update_args(args)

    trainer = IncrementalTrainer(args)
    # trainer.run()

    trainer.run()
