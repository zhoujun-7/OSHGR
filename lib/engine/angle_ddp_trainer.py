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
from ..dataset.oshgr import UnconstrainedPretrain, UnconstrainedFSSIHGR, UnconstrainedFSCIHGR
from ..dataset.sampler import DDP_CategoriesSampler, CategoriesSampler
from ..tool.visulize_fn import detach_and_draw_batch_img, draw_image_pose, save_dealed_batch
from lib.tool.hand_fn import local_pose_2_image_pose


class DDP_Trainer(BaseTrainer):
    def __init__(self, cfg):
        self.cfg = cfg
        self.SEED = self.cfg.TRAIN.SEED
        self.WORLD_SIZE = self.cfg.DDP.WORLD_SIZE

    def setup_logger(self):
        self.logger, _, final_log_file = setup_DDP_logger(self.cfg.LOG.DIR, self.RANK)
        self.cfg.LOG.DIR = final_log_file
        self.logger.ddp_info(pformat(self.cfg))

    def setup_ddp_dataset(self):
        trainset_HGR = UnconstrainedPretrain(
            DATA_DIR=self.cfg.DATASET.OSHGR_DIR,
            RESOLUTION=self.cfg.DATASET.RESOLUTION,
            DEPTH_NORMALIZE=self.cfg.DATASET.DEPTH_NORMALIZE,
            **self.cfg.AUGMENT,
        )

        trainset_HPE = BigHand(
            DATA_DIR=self.cfg.DATASET.BIGHAND_DIR,
            RESOLUTION=self.cfg.DATASET.RESOLUTION,
            DEPTH_NORMALIZE=self.cfg.DATASET.DEPTH_NORMALIZE,
            **self.cfg.AUGMENT,
        )

        trainset = ConcatDataset([trainset_HGR, trainset_HPE])
        cls_ls = trainset_HGR.label_ls.tolist() + trainset_HPE.label_ls

        num_HGR_sample = int(
            self.cfg.TRAIN.PER_GPU_BATCH
            * self.cfg.DATASET.HGR_HPE_RATIO[0]
            / (self.cfg.DATASET.HGR_HPE_RATIO[0] + self.cfg.DATASET.HGR_HPE_RATIO[1])
        )
        num_HPE_sample = self.cfg.TRAIN.PER_GPU_BATCH - num_HGR_sample

        if self.cfg.DATASET.USE_SAMPLER:
            if self.WORLD_SIZE > 1:
                self.ddp_sampler = DDP_CategoriesSampler(
                    trainset,
                    cls_ls,
                    episode=self.cfg.TRAIN.EPISODE,
                    class_other_turn=(num_HGR_sample, num_HPE_sample),
                    n_per=self.cfg.DATASET.PRETRAIN_SHOT,
                    seed=self.cfg.TRAIN.SEED,
                )

                self.dataloader = DataLoader(
                    dataset=trainset,
                    sampler=self.ddp_sampler,
                    drop_last=True,
                    num_workers=self.cfg.TRAIN.NUM_WORK,
                    batch_size=self.cfg.TRAIN.PER_GPU_BATCH,
                )
            else:
                self.ddp_sampler = CategoriesSampler(
                    label=cls_ls,
                    n_batch=self.cfg.TRAIN.EPISODE,
                    n_cls=num_HGR_sample // self.cfg.DATASET.PRETRAIN_SHOT,
                    n_per=self.cfg.DATASET.PRETRAIN_SHOT,
                    n_extra=num_HPE_sample,
                )
                self.dataloader = DataLoader(
                    dataset=trainset,
                    batch_sampler=self.ddp_sampler,
                    num_workers=self.cfg.TRAIN.NUM_WORK,
                )
        else:
            self.dataloader = DataLoader(
                dataset=trainset_HGR,
                num_workers=self.cfg.TRAIN.NUM_WORK,
                batch_size=self.cfg.TRAIN.PER_GPU_BATCH,
                shuffle=True,
                pin_memory=True,
            )

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

    def setup_model(self):
        if self.cfg.MODEL.BACKBONE == "resnet18":
            self.model = BaseNet(CLS_NUM=self.cfg.DATASET.NUM_CLASS)
        elif self.cfg.MODEL.BACKBONE == "vit":
            self.model = MyNet(
                ONLY_MAE=self.cfg.MODEL.ONLY_MAE,
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

        self.logger.ddp_info(self.model)

        if self.cfg.TRAIN.CHECKPOINT:
            self.logger.ddp_info("Loading checkpoint...")
            ckpt = torch.load(self.cfg.TRAIN.CHECKPOINT, map_location=torch.device("cpu"))
            ckpt.pop("classifier.proto.weight")
            self.model.load_state_dict(ckpt, strict=False)
        else:
            self.logger.ddp_info("No checkpoint is specified.")

        self.model.to(self.RANK)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if self.WORLD_SIZE > 1:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.RANK],
                output_device=self.RANK,
                find_unused_parameters=True,
            )
        else:
            self.ddp_model = self.model

    def setup_optimizer(self):
        if self.cfg.OPTIM.OPTIMIZER == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.OPTIM.LR,
                weight_decay=self.cfg.OPTIM.WD,
                betas=(0.9, 0.95),
            )
        elif self.cfg.OPTIM.OPTIMIZER == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.OPTIM.LR,
                weight_decay=self.cfg.OPTIM.WD,
                betas=(0.9, 0.95),
            )
        elif self.cfg.OPTIM.OPTIMIZER == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.OPTIM.LR,
                nesterov=True,
                momentum=0.9,
                weight_decay=self.cfg.OPTIM.WD,
            )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.OPTIM.LR_STEP,
            gamma=self.cfg.OPTIM.LR_FACTOR,
        )
        self.scaler = GradScaler()

    def cal_final_loss(self, loss):
        info = ""
        loss_final = loss["logit"] * self.cfg.LOSS.CLS
        info += f"  loss_class: {loss['logit']:.2e}"
        if self.cfg.MODEL.BACKBONE == "vit":
            if self.cfg.MODEL.MAE.USE:
                loss_final += loss["mae"] * self.cfg.LOSS.MAE
                info += f"  loss_MAE: {loss['mae']:.2e}"
            if self.cfg.MODEL.A2J.USE:
                loss_final += loss["uvd"] * self.cfg.LOSS.UVD
                info += f"  loss_A2J: {loss['uvd']:.2e}"
            if self.cfg.MODEL.GCN.USE:
                loss_final += loss["kpt"] * self.cfg.LOSS.KPT
                loss_final += loss["ang"] * self.cfg.LOSS.ANG
                loss_final += loss["syn"] * self.cfg.LOSS.SYN
                info += f"  loss_KPT: {loss['kpt']:.2e}"
                info += f"  loss_ANG: {loss['ang']:.2e}"
                info += f"  loss_SYN: {loss['syn']:.2e}"

        loss_info = f"loss_FINAL: {loss_final:.2e}" + info
        return loss_final, loss_info

    def incr_shape(self):
        shape_ls = ["Glove_Thin", "Glove_Thick", "Glove_Half-finger"]
        shape_acc_dt = {}
        for shape in shape_ls:
            incr_shape_dataset = UnconstrainedFSSIHGR(
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
        test_shape_dataset = UnconstrainedFSSIHGR(
            PHASE="test",
            INCR_SHAPE=shape,
        )
        test_shape_dataset.label_ls += self.cfg.DATASET.NUM_BASE_CLASS

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
            for image, noise_image, label, pose, angle, _ in tqdm.tqdm(test_shape_dataloader, desc=f"test shape {shape}"):
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
        return acc

    def incr_class(self):
        self.model.classifier.restore_proto(is_kaiming=True)
        for sess in range(1, self.cfg.DATASET.NUM_SESSION + 1):
            incr_class_dataset = UnconstrainedFSCIHGR(
                PHASE="incremental",
                SESSION=sess,
                INCR_WAY=self.cfg.DATASET.INCR_WAY,
                INCR_SHOT=self.cfg.DATASET.INCR_SHOT,
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
        test_class_dataset = UnconstrainedFSCIHGR(
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
            for image, noise_image, label, pose, angle, _ in tqdm.tqdm(
                test_class_dataloader, desc="test incremental class"
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

    def log_image(self, x_dt, pd_dt, gt_dt):
        log_img = []
        log_img.append(detach_and_draw_batch_img(x_dt["image"], gt_dt["uvd"], "manopth"))
        if self.cfg.MODEL.A2J.USE:
            log_img.append(detach_and_draw_batch_img(x_dt["image"], pd_dt["uvd"], "manopth"))
        if self.cfg.MODEL.GCN.USE:
            log_img.append(detach_and_draw_batch_img(x_dt["image"], pd_dt["kpt"], "manopth"))
            log_img.append(draw_image_pose(local_pose_2_image_pose(pd_dt["ang2kpt"][1]), background="white"))
            log_img.append(draw_image_pose(local_pose_2_image_pose(pd_dt["ang2kpt"][0]), background=0.8))
        if self.cfg.MODEL.MAE.USE:
            log_img.append(detach_and_draw_batch_img(x_dt["noise"]))
            log_img.append(detach_and_draw_batch_img(pd_dt["mae"]))
        save_dealed_batch(log_img, 2 * len(log_img), os.path.join(self.cfg.LOG.DIR, "pred.jpg"))

    def pretrain(self):
        best_acc = -1
        loop_time_counter = MultiLoopTimeCounter(
            [
                self.cfg.TRAIN.EPOCH,
                self.cfg.TRAIN.EPISODE,
            ]
        )

        for epoch in range(self.cfg.TRAIN.EPOCH):
            self.ddp_sampler.set_epoch(epoch) if self.cfg.DATASET.USE_SAMPLER else None
            self.model.classifier.restore_proto(is_kaiming=True)

            self.logger.ddp_info(f"Train Epoch: {epoch}")
            self.model.train(True)
            self.logger.ddp_info(f"Current LR: {self.scheduler.get_last_lr()[0]}")
            for epi, (image, noise_image, label, pose, angle, _) in enumerate(self.dataloader):
                with torch.autocast(device_type="cuda", dtype=self.cfg.TRAIN.DTYPE):
                    x_dt = {
                        "image": image.to(self.RANK),
                        "noise": noise_image.to(self.RANK),
                    }
                    gt_dt = {
                        "class": label.to(self.RANK),
                        "uvd": pose.to(self.RANK),
                        "ang": angle[:, :15].to(self.RANK),
                    }

                    pd_dt, loss_dt = self.ddp_model(x_dt, gt_dt, "pretrain")

                    loss_final, loss_info = self.cal_final_loss(loss_dt)

                loss_final = loss_final / self.cfg.OPTIM.GRAD_ACCM
                self.scaler.scale(loss_final).backward()

                if epi % self.cfg.OPTIM.GRAD_ACCM == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                torch.cuda.synchronize(device=self.RANK)

                all_time, accm_time, _ = loop_time_counter.step()
                if self.RANK == 0 and epi % self.cfg.LOG.PRINT_FREQ == 0:
                    acc = count_acc(pd_dt["logit"], gt_dt["class"].to(self.DEVICE))

                    info = (
                        f"[{epoch:02d}/{self.cfg.TRAIN.EPOCH}][{epi:03d}/{self.cfg.TRAIN.EPISODE}]"
                        + f" | time: epo[{accm_time[0]/3600:.2f}/{all_time[0]/3600:.2f}h] epi[{accm_time[1]/60:.1f}/{all_time[1]/60:.1f}m] | "
                        + loss_info
                        + f" | acc: {acc:.2f}"
                    )

                    self.logger.ddp_info(info)
                    self.log_image(x_dt, pd_dt, gt_dt)

                # break
            self.scheduler.step()

            self.model.train(False)

            if self.cfg.TRAIN.DATA_INIT:
                self.logger.ddp_info(f"average base prototype...")
                self.model.incremental_training(self.update_base_loader)

            self.model.classifier.save_proto()

            if self.RANK == 0:
                if self.cfg.TRAIN.VALIDATION:
                    self.logger.ddp_info(f"incremental shape")
                    shape_acc_dt = self.incr_shape()
                    self.logger.ddp_info(pformat(shape_acc_dt))

                    self.logger.ddp_info(f"incremental class")
                    class_acc_dt = self.incr_class()
                    # class_acc_dt = self.test_class(9)
                    self.logger.ddp_info(pformat(class_acc_dt))

                    if class_acc_dt[9] > best_acc:
                        self.logger.ddp_info(f"Find better model.")
                        best_acc = class_acc_dt[9]

                        self.model.classifier.restore_proto(is_kaiming=True)
                        os.system(f"rm {self.cfg.LOG.DIR}" + "/ckpt-best-*.pth") if epoch > 0 else None
                        torch.save(
                            self.model.state_dict(),
                            self.cfg.LOG.DIR + f"/ckpt-best-{epoch}-{best_acc}.pth",
                        )
                self.model.classifier.restore_proto(is_kaiming=True)
                os.system(f"rm {self.cfg.LOG.DIR}" + "/ckpt-last-*.pth") if epoch > 0 else None
                torch.save(self.model.state_dict(), self.cfg.LOG.DIR + f"/ckpt-last-{epoch}.pth")

            if self.cfg.TRAIN.SAVE_EACH:
                torch.save(self.model.state_dict(), self.cfg.LOG.DIR + f"/ckpt-{epoch}.pth")

    def ddp_worker(self, rank=0):
        self.RANK = rank
        self.DEVICE = rank
        self.setup_seed(self.SEED + self.RANK)
        self.setup_logger()
        self.setup_ddp_env(rank, port=str(self.cfg.DDP.MASTER_PORT))
        self.setup_ddp_dataset()
        self.setup_model()
        self.setup_optimizer()
        self.pretrain()
        self.clean()
