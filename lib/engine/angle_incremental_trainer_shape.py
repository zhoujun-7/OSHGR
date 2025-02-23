import os
import time
import tqdm
import random
import numpy as np
import torch
from pprint import pformat
import torch.nn as nn
import torch.distributed as distributed
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data.distributed import DistributedSampler

from .base_trainer import BaseTrainer
from ..model.angle_build import BaseNet, MyNet
from ..tool.misc import setup_DDP_logger, MultiLoopTimeCounter, count_acc
from ..dataset.bighand import BigHand
from ..dataset.oshgr import (
    UnconstrainedPretrain,
    UnconstrainedFSSIHGR,
    UnconstrainedFSCIHGR,
    ConstrainedFSSIHGR,
    ConstrainedFSCIHGR,
)
from ..dataset.oshgr_shape import UnconstrainedFSSIHGR
from ..dataset.sampler import DDP_CategoriesSampler, CategoriesSampler
from ..tool.visulize_fn import detach_and_draw_batch_img, draw_image_pose, save_dealed_batch
from lib.tool.hand_fn import local_pose_2_image_pose


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
        shape_ls = ["Bare", "Glove_Thin", "Glove_Half-finger", "Glove_Thick"]

        shape_acc_dt = {}
        shape_test_set = []

        from collections import defaultdict
        save_dt = defaultdict(list)

        for i, shape in enumerate(shape_ls):
            print(shape)

            if shape != "Bare":
                incr_shape_dataset = UnconstrainedFSSIHGR(
                    PHASE="incremental",
                    INCR_SHAPE=shape,
                    INCR_SHOT=self.cfg.DATASET.INCR_SHOT,
                    IS_AUGMENT=self.cfg.AUGMENT.INCR_AUGMENT,
                )
                incr_shape_dataset.label_ls += self.cfg.DATASET.NUM_BASE_CLASS * (i)

                incr_shape_dataloader = DataLoader(
                    dataset=incr_shape_dataset,
                    num_workers=0,
                    batch_size=64,
                    shuffle=False,
                )

                self.model.incremental_training(incr_shape_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)

                test_shape_dataset = UnconstrainedFSSIHGR(
                    PHASE="test",
                    INCR_SHAPE=shape,
                )
                test_shape_dataset.label_ls += self.cfg.DATASET.NUM_BASE_CLASS * (i)
                print(np.unique(test_shape_dataset.label_ls))

            else:
                test_shape_dataset = UnconstrainedFSCIHGR(
                    PHASE="test",
                    SESSION=0,
                )


            print("classifier number: ", self.model.classifier.proto.weight.shape[0])


            shape_test_set.append(test_shape_dataset)
            # test_cat_dataset = ConcatDataset(shape_test_set)
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
                for image, noise_image, label, pose, angle, timestamp in tqdm.tqdm(
                    test_shape_dataloader, desc=f"{len(test_cat_dataset)}"
                ):
                    x_dt = {"image": image.to(self.RANK)}
                    gt_dt = {"class": label.to(self.RANK)}
                    pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test")
                    all_logit.append(pd_dt["logit"])
                    all_label.append(gt_dt["class"])

                #     # region [tmp]
                #     if shape == "Glove_Thick":
                #         save_dt["emb"].append(pd_dt["emb"][:, :1920].clone().detach().cpu().numpy())
                #         save_dt["label"].append(label.clone().detach().cpu().numpy())
                #         save_dt["image"].append(image.clone().detach().cpu().numpy())

                # if shape == "Glove_Thick":
                #     save_path = "out/tmm_review/hand-shape_emb-label-image.pkl"
                #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                #     import pickle
                #     with open(save_path, "wb") as f:
                #         pickle.dump(save_dt, f)
                #     # endregion [tmp]



            all_logit = torch.cat(all_logit)
            all_label = torch.cat(all_label)

            # all_pred = torch.argmax(all_logit[:, :23], dim=1)
            all_pred = torch.argmax(all_logit, dim=1)
            all_pred = all_pred % self.cfg.DATASET.NUM_BASE_CLASS
            all_label = all_label % self.cfg.DATASET.NUM_BASE_CLASS

            acc = (all_pred == all_label).to(torch.float32).mean().item() * 100
            acc = float(f"{acc:.2f}")
            shape_acc_dt[shape] = acc

            print(f"{shape}: ", acc)

            # region [extend finger]
            unextend_gesture = [1, 3, 5, 12, 13, 14, 17, 18, 22]
            extend_gesture = [2, 4, 6, 7, 8, 9, 10, 11, 15, 16, 19, 20, 21, 23]

            is_extend = []
            for all_label_i in all_label:
                if all_label_i+1 in unextend_gesture:
                    is_extend.append(False)
                elif all_label_i+1 in extend_gesture:
                    is_extend.append(True)
                else:
                    raise ValueError("gesture not in extend or unextend gesture")
            is_extend = torch.tensor(is_extend).to(self.RANK)

            extend_label = all_label[is_extend]
            extend_pred = all_pred[is_extend]
            extend_acc = (extend_label == extend_pred).to(torch.float32).mean().item() * 100

            unextend_label = all_label[~is_extend]
            unextend_pred = all_pred[~is_extend]
            unextend_acc = (unextend_label == unextend_pred).to(torch.float32).mean().item() * 100

            extend_acc = float(f"{extend_acc:.2f}")
            unextend_acc = float(f"{unextend_acc:.2f}")

            print(f'extend num: {extend_label.shape[0]}, unextend num: {unextend_label.shape[0]}')
            print(f"extend: {extend_acc}, unextend: {unextend_acc}")
            # endregion [extend finger]
        
        return shape_acc_dt


    def run(self):
        self.setup_logger()
        self.setup_model()

        self.model.train(False)
        update_base_dataset = UnconstrainedPretrain(
            DATA_DIR=self.cfg.DATASET.OSHGR_DIR,
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
