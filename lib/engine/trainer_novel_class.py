import tqdm
import numpy as np
import torch
from pprint import pformat
from torch.utils.data import DataLoader, ConcatDataset

from ..model.angle_build import BaseNet, MyNet
from ..tool.misc import setup_DDP_logger
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

    def incr_class(self):

        self.model.classifier.restore_proto(is_kaiming=True)
        for sess in range(1, self.cfg.DATASET.NUM_SESSION + 1):
            start_label = 23 + self.cfg.DATASET.INCR_WAY * (sess - 1)
            incr_class_dataset = ClassDataset(
                DATA_DIR="data/OHG_cropped/TrainingNovelClass",
                LABELS=list(range(start_label, start_label + self.cfg.DATASET.INCR_WAY)),
                IS_AUGMENT=self.cfg.AUGMENT.INCR_AUGMENT,
            )

            incr_class_dataloader = DataLoader(
                dataset=incr_class_dataset,
                num_workers=0,
                batch_size=64,
                shuffle=False,
            )
            self.model.incremental_training(incr_class_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)
        class_acc_dt = self.test_class(sess)

        return class_acc_dt


    def test_class(self, session):
        test_class_dataset = ClassDataset(
            DATA_DIR="data/OHG_cropped/EvaluationClass",
            LABELS=list(range(23 + self.cfg.DATASET.INCR_WAY * session)),
            IS_AUGMENT=False,
        )

        test_class_dataloader = DataLoader(
            dataset=test_class_dataset,
            num_workers=self.cfg.TRAIN.NUM_WORK,
            pin_memory=True,
            batch_size=64,
            shuffle=False,
        )

        all_logit = []
        all_label = []
        timestamp_ls = []
        with torch.no_grad():
            for image, noise_image, label, pose, angle, timestamp in tqdm.tqdm(
                test_class_dataloader, desc=f"test incremental class {len(test_class_dataset)}"
            ):
                x_dt = {"image": image.to(self.RANK)}
                gt_dt = {"class": label.to(self.RANK)}
                pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test", save=True)
                all_logit.append(pd_dt["logit"])
                all_label.append(gt_dt["class"])

                timestamp_ls.append(timestamp)


        all_logit = torch.cat(all_logit)
        all_label = torch.cat(all_label)

        class_acc_dt = {}
        for sess in range(session + 1):
            num_test_class = self.cfg.DATASET.NUM_BASE_CLASS + self.cfg.DATASET.INCR_WAY * sess
            m = all_label < num_test_class

            sess_label = all_label[m]
            sess_pred = torch.argmax(all_logit[m][:, :num_test_class], dim=1)

            sess_acc = (sess_pred == sess_label).to(torch.float32).mean().item() * 100
            sess_acc = float(f"{sess_acc:.2f}")
            class_acc_dt[sess] = sess_acc

        return class_acc_dt


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


        class_acc_dt = self.incr_class()
        self.logger.ddp_info(pformat(class_acc_dt))

