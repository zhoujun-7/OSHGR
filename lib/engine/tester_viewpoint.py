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
    ViewPointRPY,
    ViewPointRPY_Incr,
)
from ..dataset.sampler import DDP_CategoriesSampler, CategoriesSampler
from ..tool.visulize_fn import detach_and_draw_batch_img, draw_image_pose, save_dealed_batch
from lib.tool.hand_fn import local_pose_2_image_pose


class ViewPointTester:
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
        if self.cfg.TRAIN.TASK == "CFSSI":
            FSSIHGR = ConstrainedFSSIHGR
        elif self.cfg.TRAIN.TASK == "UFSSI":
            FSSIHGR = UnconstrainedFSSIHGR

        shape_ls = ["Glove_Thin", "Glove_Thick", "Glove_Half-finger"]
        shape_acc_dt = {}
        for shape in shape_ls:
            incr_shape_dataset = FSSIHGR(
                PHASE="incremental",
                INCR_SHAPE=shape,
                INCR_SHOT=self.cfg.DATASET.INCR_SHOT,
                IS_AUGMENT=self.cfg.AUGMENT.INCR_AUGMENT,
            )
            incr_shape_dataset.label_ls += self.cfg.DATASET.NUM_BASE_CLASS

            incr_shape_dataloader = DataLoader(
                dataset=incr_shape_dataset,
                num_workers=0,
                batch_size=64,
                shuffle=False,
            )

            self.model.classifier.restore_proto(is_kaiming=True)
            self.model.incremental_training(incr_shape_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)
            acc = self.test_shape(shape)
            shape_acc_dt[shape] = acc

        return shape_acc_dt

    def test_shape(self, shape):
        if self.cfg.TRAIN.TASK == "CFSSI":
            FSSIHGR = ConstrainedFSSIHGR
        elif self.cfg.TRAIN.TASK == "UFSSI":
            FSSIHGR = UnconstrainedFSSIHGR

        test_shape_dataset = FSSIHGR(
            PHASE="test",
            INCR_SHAPE=shape,
        )
        # test_shape_dataset.label_ls += self.cfg.DATASET.NUM_BASE_CLASS

        test_shape_dataloader = DataLoader(
            dataset=test_shape_dataset,
            num_workers=0,
            pin_memory=True,
            batch_size=64,
            shuffle=False,
        )

        all_logit = []
        all_label = []
        with torch.no_grad():
            for image, noise_image, label, pose, angle in tqdm.tqdm(
                test_shape_dataloader, desc=f"test shape {shape} {len(test_shape_dataset)}"
            ):
                x_dt = {"image": image.to(self.RANK)}
                gt_dt = {"class": label.to(self.RANK)}
                pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test")
                all_logit.append(pd_dt["logit"])
                all_label.append(gt_dt["class"])

        all_logit = torch.cat(all_logit)
        all_label = torch.cat(all_label)

        all_pred = torch.argmax(all_logit, dim=1)
        all_pred = all_pred % self.cfg.DATASET.NUM_BASE_CLASS
        all_label = all_label % self.cfg.DATASET.NUM_BASE_CLASS

        acc = (all_pred == all_label).to(torch.float32).mean().item() * 100
        acc = float(f"{acc:.2f}")

        # print(all_label)
        # print(all_pred)
        # print(all_label != all_pred)
        # print(acc)

        # print(all_logit[all_label != all_pred])
        # exit()

        return acc

    # def incr_class(self, VIEW_TYPE):
    #     if self.cfg.TRAIN.TASK == "CFSCI":
    #         FSCIHGR = ConstrainedFSCIHGR
    #     elif self.cfg.TRAIN.TASK == "UFSCI":
    #         FSCIHGR = UnconstrainedFSCIHGR

    #     incr_class_dataset = ViewPointRPY_Incr(TYPE=VIEW_TYPE, SESSION=9, IS_AUGMENT=False)

    #     self.model.classifier.restore_proto(is_kaiming=True)
    #     # for sess in range(1, self.cfg.DATASET.NUM_SESSION + 1):
    #     #     incr_class_dataset = FSCIHGR(
    #     #         PHASE="incremental",
    #     #         SESSION=sess,
    #     #         INCR_WAY=self.cfg.DATASET.INCR_WAY,
    #     #         INCR_SHOT=self.cfg.DATASET.INCR_SHOT,
    #     #         IS_AUGMENT=self.cfg.AUGMENT.INCR_AUGMENT,
    #     #     )
    #     #     print(len(incr_class_dataset))
    #     incr_class_dataloader = DataLoader(
    #         dataset=incr_class_dataset,
    #         num_workers=0,
    #         batch_size=64,
    #         shuffle=False,
    #     )
        
    #     print(len(incr_class_dataset))
    #     self.model.incremental_training(incr_class_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)

    #     # class_acc_dt = self.test_class(9)

    #     # return class_acc_dt


    def incr_class(self, _):
        FSCIHGR = UnconstrainedFSCIHGR

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

        self.model.classifier.restore_proto(is_kaiming=True)
        for sess in range(1, self.cfg.DATASET.NUM_SESSION + 1):
            incr_class_dataset = FSCIHGR(
                PHASE="incremental",
                SESSION=sess,
                INCR_WAY=self.cfg.DATASET.INCR_WAY,
                INCR_SHOT=1,
                IS_AUGMENT=self.cfg.AUGMENT.INCR_AUGMENT,
            )
            incr_class_dataloader = DataLoader(
                dataset=incr_class_dataset,
                num_workers=0,
                batch_size=64,
                shuffle=False,
            )
            self.model.incremental_training(incr_class_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)

        # class_acc_dt = self.test_class(sess)

        # return class_acc_dt


    def test_class(self, session):
        if self.cfg.TRAIN.TASK == "CFSCI":
            FSCIHGR = ConstrainedFSCIHGR
        elif self.cfg.TRAIN.TASK == "UFSCI":
            FSCIHGR = UnconstrainedFSCIHGR

        test_class_dataset = FSCIHGR(
            PHASE="test",
            SESSION=session,
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
        with torch.no_grad():
            for image, noise_image, label, pose, angle in tqdm.tqdm(
                test_class_dataloader, desc=f"test incremental class {len(test_class_dataset)}"
            ):
                x_dt = {"image": image.to(self.RANK)}
                gt_dt = {"class": label.to(self.RANK)}
                pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test")
                all_logit.append(pd_dt["logit"])
                all_label.append(gt_dt["class"])

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

        # update_base_dataset = UnconstrainedPretrain(
        #     DATA_DIR=self.cfg.DATASET.OSHGR_DIR,
        #     RESOLUTION=self.cfg.DATASET.RESOLUTION,
        #     DEPTH_NORMALIZE=self.cfg.DATASET.DEPTH_NORMALIZE,
        #     IS_AUGMENT=False,
        # )
        # self.update_base_loader = DataLoader(
        #     dataset=update_base_dataset,
        #     num_workers=self.cfg.TRAIN.NUM_WORK,
        #     pin_memory=True,
        #     batch_size=self.cfg.TRAIN.PER_GPU_BATCH,
        #     shuffle=False,
        # )
        # self.model.incremental_training(self.update_base_loader)

        # class_acc_dt = self.test_class(0)
        # print(class_acc_dt)
        # exit()

        self.cfg.TRAIN.TASK = "CFSCI"
        self.cfg.DATASET.INCR_SHOT = 5
        # VIEW_TYPE = "Left_to_Right"
        VIEW_TYPE = "Up_to_Down"

        self.cfg.DATASET.NUM_SESSION = 4
        class_acc_dt = self.incr_class(VIEW_TYPE)
        self.logger.ddp_info(pformat(class_acc_dt))

        testset = ViewPointRPY(TYPE=VIEW_TYPE, SESSION=4, IS_AUGMENT=False)
        testloader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=self.cfg.TRAIN.PER_GPU_BATCH,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        all_logit = []
        all_label = []
        viewpoint_ls = []
        timestamp_ls = []
        with torch.no_grad():
            for image, noise_image, label, view_point, timestamp in tqdm.tqdm(
                testloader, desc=f"test {VIEW_TYPE} class {len(testloader)}"
            ):
                x_dt = {"image": image.to(self.RANK)}
                gt_dt = {"class": label.to(self.RANK)}
                pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test")
                all_logit.append(pd_dt["logit"])
                all_label.append(gt_dt["class"])
                viewpoint_ls.append(view_point)
                timestamp_ls.extend(timestamp)

        all_logit = torch.cat(all_logit)
        all_label = torch.cat(all_label)
        all_viewpoint = torch.cat(viewpoint_ls)

        all_pred = torch.argmax(all_logit, dim=1)

        all_pred = all_pred == all_label

        for i in torch.unique(all_label):
            m = all_label == i
            print(i, all_pred[m].to(torch.float32).mean())


        print(all_pred.to(torch.float32).mean())

        print(all_viewpoint.shape)
        print(all_pred.shape)

        # np.save(
        #     f"log/Train/Ours/best/OUR_{VIEW_TYPE}_prediction_{self.cfg.DATASET.INCR_SHOT}",
        #     all_pred.cpu().numpy(),
        # )
        # np.save(
        #     f"log/Train/Ours/best/OUR_{VIEW_TYPE}_view_point_{self.cfg.DATASET.INCR_SHOT}",
        #     all_viewpoint.cpu().numpy(),
        # )

        import pickle
        dt = {"time": timestamp_ls, "acc": all_pred.cpu().numpy()}
        print(len(timestamp_ls))
        print(len(all_pred))
        with open("./time_n_acc_pitch20.pkl", 'wb') as f:
            pickle.dump(dt, f)
