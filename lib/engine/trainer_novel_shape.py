import tqdm
import torch
from pprint import pformat
from torch.utils.data import DataLoader, ConcatDataset

from ..model.angle_build import BaseNet, MyNet
from ..tool.misc import setup_DDP_logger
from ..dataset.novel_shape import ShapeDataset
from ..dataset.novel_class import ClassDataset


class IncrementalTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.SEED = self.cfg.TRAIN.SEED
        self.WORLD_SIZE = 1
        self.RANK = 0

    def setup_logger(self):
        self.logger, _, final_log_file = setup_DDP_logger(self.cfg.LOG.DIR, self.RANK)
        self.cfg.LOG.DIR = final_log_file
        self.logger.ddp_info(pformat(self.cfg))

    def setup_model(self):
        if self.cfg.MODEL.BACKBONE == "resnet18":
            self.model = BaseNet(CLS_NUM=self.cfg.DATASET.NUM_CLASS)
        elif self.cfg.MODEL.BACKBONE == "vit":
            self.model = MyNet(
                MAE_USE=self.cfg.MODEL.MAE.USE,
                MAE_FROM_NOISE=self.cfg.MODEL.MAE.FROM_NOISE,
                A2J_USE=self.cfg.MODEL.A2J.USE,
                A2J_LOC=self.cfg.MODEL.A2J.LOC,
                A2J_DETACH=self.cfg.MODEL.A2J.DETACH,
                DOWNSAMPLE=self.cfg.MODEL.DOWNSAMPLE,
                GCN_USE=self.cfg.MODEL.GCN.USE,
                GCN_IN_DIM=self.cfg.MODEL.GCN.IN_DIM,
                GCN_HID_DIM=self.cfg.MODEL.GCN.HID_DIM,
                CLS_NUM=self.cfg.DATASET.NUM_CLASS,
                NUM_BASE_CLASS=self.cfg.DATASET.NUM_BASE_CLASS,
            )

        if self.cfg.TRAIN.CHECKPOINT:
            self.logger.ddp_info("Loading checkpoint...")
            ckpt = torch.load(self.cfg.TRAIN.CHECKPOINT, map_location=torch.device("cpu"))
            ckpt.pop("classifier.proto.weight")
            self.model.load_state_dict(ckpt, strict=False)
        else:
            self.logger.ddp_info("No checkpoint is specified.")

        self.model.to(self.RANK)

    def incr_shape(self):
        shape_ls = ["Bare", "Thin", "Half-finger", "Thick"]

        shape_acc_dt = {}
        shape_test_set = []
        for i, shape in enumerate(shape_ls):
            print(shape)

            if shape != "Bare":
                incr_shape_dataset = ShapeDataset(
                    INCR_SHAPE=shape,
                    DATA_DIR="data/OHG_cropped/TrainingNovelShape",
                    IS_AUGMENT=self.cfg.AUGMENT.INCR_AUGMENT,
                )

                incr_shape_dataset.label_ls += self.cfg.DATASET.NUM_BASE_CLASS * (i + 1)

                incr_shape_dataloader = DataLoader(
                    dataset=incr_shape_dataset,
                    num_workers=0,
                    batch_size=64,
                    shuffle=False,
                )

                self.model.incremental_training(incr_shape_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)

                test_shape_dataset = ShapeDataset(
                    INCR_SHAPE=shape,
                    DATA_DIR="data/OHG_cropped/EvaluationShape",
                )

            else:
                test_shape_dataset = ClassDataset(
                    DATA_DIR="data/OHG_cropped/EvaluationClass",
                    LABELS=list(range(23)),
                    IS_AUGMENT=False,
                )


            print("classifier number: ", self.model.classifier.proto.weight.shape[0])

            shape_test_set.append(test_shape_dataset)
            test_cat_dataset = ConcatDataset(shape_test_set)
            test_shape_dataloader = DataLoader(
                dataset=test_cat_dataset,
                num_workers=0,
                pin_memory=True,
                batch_size=64,
                shuffle=False,
            )

            all_logit = []
            all_label = []

            with torch.no_grad():
                for image, noise_image, label, pose, angle, _ in tqdm.tqdm(
                    test_shape_dataloader, desc=f"{len(test_cat_dataset)}"
                ):
                    x_dt = {"image": image.to(self.RANK)}
                    gt_dt = {"class": label.to(self.RANK)}
                    pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test")
                    all_logit.append(pd_dt["logit"])
                    all_label.append(gt_dt["class"])
                    
                    print(pd_dt["logit"][:3])
                    print(gt_dt["class"][:3])

            all_logit = torch.cat(all_logit)
            all_label = torch.cat(all_label)

            all_pred = torch.argmax(all_logit, dim=1)
            all_pred = all_pred % self.cfg.DATASET.NUM_BASE_CLASS
            all_label = all_label % self.cfg.DATASET.NUM_BASE_CLASS

            acc = (all_pred == all_label).to(torch.float32).mean().item() * 100
            acc = float(f"{acc:.2f}")
            shape_acc_dt[shape] = acc

        return shape_acc_dt


    def run(self):
        self.setup_logger()
        self.setup_model()

        self.model.train(False)

        update_base_dataset = ClassDataset(
            DATA_DIR="data/OHG_cropped/TrainingBase",
            LABELS=list(range(23)),
            RESOLUTION=self.cfg.DATASET.RESOLUTION,
            DEPTH_NORMALIZE=self.cfg.DATASET.DEPTH_NORMALIZE,
            IS_AUGMENT=False,
        )
        self.update_base_loader = DataLoader(
            dataset=update_base_dataset,
            num_workers=self.cfg.TRAIN.NUM_WORK,
            pin_memory=True,
            batch_size=self.cfg.TRAIN.PER_GPU_BATCH,
            shuffle=False,
        )
        self.model.incremental_training(self.update_base_loader)


        shape_acc_dt = self.incr_shape()
        self.logger.ddp_info(pformat(shape_acc_dt))
