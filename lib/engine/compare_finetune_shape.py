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
from ..model.build import BaseNet, MyNet
from ..tool.misc import setup_DDP_logger, MultiLoopTimeCounter, count_acc
from ..dataset.bighand import BigHand
from ..dataset.oshgr import UnconstrainedPretrain, UnconstrainedFSSIHGR, UnconstrainedFSCIHGR, ConstrainedFSSIHGR, ConstrainedFSCIHGR
from ..dataset.sampler import DDP_CategoriesSampler, CategoriesSampler
from ..tool.visulize_fn import detach_and_draw_batch_img, draw_image_pose, save_dealed_batch
from lib.tool.hand_fn import local_pose_2_image_pose
from ..model.build_finetune import net
from ..dataset.oshgr_shape import UnconstrainedFSSIHGR, get_dataset

class Finetune_Trainer(BaseTrainer):
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
        self.model = net
        self.logger.ddp_info(self.model)

        if self.cfg.TRAIN.CHECKPOINT:
            self.logger.ddp_info("Loading checkpoint...")
            ckpt = torch.load(self.cfg.TRAIN.CHECKPOINT, map_location=torch.device("cpu"))
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
                self.model.fc.parameters(),
                lr=0.003,
                # weight_decay=self.cfg.OPTIM.WD,
                # betas=(0.9, 0.95),
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
        loss_info = f"loss_FINAL: {loss_final:.2e}" + info
        return loss_final, loss_info

    def incr_shape(self):
        # self.model.fc.weight.data = torch.cat([self.model.fc.weight.data, torch.randn_like(self.model.fc.weight.data)[:(23*5-68)]])
        # self.model.fc.bias.data = torch.cat([self.model.fc.bias.data, torch.randn_like(self.model.fc.bias.data)[:(23*5-68)]])
        self.model.train(False)
        # extra_weight = self.model.fc.weight.data[:23].clone().detach()
        # extra_weight = torch.tile(extra_weight[:23], [4, 1])
        # extra_bias = self.model.fc.bias.data[:23].clone().detach()
        # extra_bias = torch.tile(extra_bias, [4])
        # self.model.fc.weight.data = torch.cat([self.model.fc.weight.data[:23], extra_weight])
        # self.model.fc.bias.data = torch.cat([self.model.fc.bias.data[:23], extra_bias])

        print("model classifier number: ", self.model.fc.weight.shape[0])
        # FSSIHGR = ConstrainedFSSIHGR
        FSSIHGR = UnconstrainedFSSIHGR
        shape_ls = ["Bare", "Glove_Thin", "Glove_Half-finger", "Glove_Thick"]
        shape_acc_dt = {}
        shape_test_ls = []
        for i in range(4):
            train_set, test_set = get_dataset(i)

            if i != 0:
                incr_shape_dataloader = DataLoader(
                    dataset=train_set,
                    num_workers=0,
                    batch_size=256,
                    shuffle=False,
                )
                
                self.model.train()
                for epoch in tqdm.trange(4000):
                    for epi, (image, noise_image, label, pose, angle, _) in enumerate(incr_shape_dataloader):
                        with torch.autocast(device_type="cuda", dtype=self.cfg.TRAIN.DTYPE):
                            image = image.to(self.RANK)
                            label = label.to(self.RANK)

                            y = self.ddp_model(image.to(self.RANK))
                            y = torch.softmax(y, dim=-1)
                            loss_final = torch.nn.functional.cross_entropy(y, label)

                        loss_final = loss_final
                        loss_final.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            
            self.model.train(False)
            test_shape_dataloader = DataLoader(
                dataset=test_set,
                num_workers=0,
                pin_memory=True,
                batch_size=256,
                shuffle=False,
            )

            print('loader length', len(test_shape_dataloader))

            all_logit = []
            all_label = []
            with torch.no_grad():
                for image, noise_image, label, pose, angle, _ in tqdm.tqdm(
                    test_shape_dataloader, desc=f"test shape {shape_ls[i]} {len(test_set)}"
                ):
                    label = label.to(self.RANK)
                    y = self.ddp_model(image.to(self.RANK))
                    y = torch.softmax(y[:, :23*(i+1)], dim=-1)

                    all_logit.append(y)
                    all_label.append(label)

            all_logit = torch.cat(all_logit)
            all_label = torch.cat(all_label)

            all_pred = torch.argmax(all_logit, dim=1)
            all_pred = all_pred % self.cfg.DATASET.NUM_BASE_CLASS
            all_label = all_label % self.cfg.DATASET.NUM_BASE_CLASS

            # region [extend]
            # label_u = all_label % 23
            # unextend_gesture = [1, 3, 5, 12, 13, 14, 17, 18, 22]
            # extend_gesture = [2, 4, 6, 7, 8, 9, 10, 11, 15, 16, 19, 20, 21, 23]
            # is_extend = []
            # for test_label_i in label_u:
            #     if test_label_i+1 in unextend_gesture:
            #         is_extend.append(False)
            #     elif test_label_i+1 in extend_gesture:
            #         is_extend.append(True)
            # is_extend = torch.tensor(is_extend).cuda() # {0: 89.92, 1: 88.98, 2: 82.56, 3: 82.4}
            # is_extend = ~is_extend # {0: 90.25, 1: 89.74, 2: 72.06, 3: 71.89}
            # if is_extend.sum() == 0:
            #     continue
            # all_pred = all_pred[is_extend] 
            # all_label = all_label[is_extend]
            # endregion [extend]

            acc = (all_pred == all_label).to(torch.float32).mean().item() * 100
            acc = float(f"{acc:.2f}")

            shape_acc_dt[i] = acc

            print(shape_ls[i], acc)

        return shape_acc_dt

    def test_shape(self, shape_ls):
        self.model.train(False)
        # FSSIHGR = ConstrainedFSSIHGR
        FSSIHGR = UnconstrainedFSSIHGR

        dataset_ls = []
        for shape in shape_ls:
            test_shape_dataset = FSSIHGR(
                PHASE="test",
                INCR_SHAPE=shape,
            )
            print('dataset length', shape, len(test_shape_dataset))
            dataset_ls.append(test_shape_dataset)
        test_shape_dataset = ConcatDataset(dataset_ls)

        test_shape_dataloader = DataLoader(
            dataset=test_shape_dataset,
            num_workers=0,
            pin_memory=True,
            batch_size=64,
            shuffle=False,
        )

        print('loader length', len(test_shape_dataloader))

        all_logit = []
        all_label = []
        with torch.no_grad():
            for image, noise_image, label, pose, angle in tqdm.tqdm(
                test_shape_dataloader, desc=f"test shape {shape_ls[-1]} {len(test_shape_dataset)}"
            ):
                label = label.to(self.RANK)
                y = self.ddp_model(image.to(self.RANK))
                y = torch.softmax(y, dim=-1)

                all_logit.append(y)
                all_label.append(label)

        all_logit = torch.cat(all_logit)
        all_label = torch.cat(all_label)

        all_pred = torch.argmax(all_logit, dim=1)
        all_pred = all_pred % self.cfg.DATASET.NUM_BASE_CLASS
        all_label = all_label % self.cfg.DATASET.NUM_BASE_CLASS

        acc = (all_pred == all_label).to(torch.float32).mean().item() * 100
        acc = float(f"{acc:.2f}")
        return acc

    def incr_class(self):
        acc_dt = {}
        FSCIHGR = UnconstrainedFSCIHGR
        # FSCIHGR = ConstrainedFSCIHGR

        print(self.test_class(0))
        
        for sess in range(1, self.cfg.DATASET.NUM_SESSION + 1):
            self.model.train()
            incr_class_dataset = FSCIHGR(
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

            for epoch in range(600):
                for epi, (image, noise_image, label, pose, angle) in enumerate(incr_class_dataloader):
                    with torch.autocast(device_type="cuda", dtype=self.cfg.TRAIN.DTYPE):
                        image = image.to(self.RANK)
                        label = label.to(self.RANK)

                        y = self.ddp_model(image.to(self.RANK))
                        y = torch.softmax(y, dim=-1)
                        loss_final = torch.nn.functional.cross_entropy(y, label)
                        loss_info = f'Loss: {loss_final.item()}'

                    loss_final = loss_final / self.cfg.OPTIM.GRAD_ACCM
                    loss_final.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            acc_dt[sess] = self.test_class(sess)
            print(sess, acc_dt[sess])
        print(acc_dt)
        return acc_dt

    def test_class(self, session):
        self.model.train(False)
        FSCIHGR = UnconstrainedFSCIHGR
        # FSCIHGR = ConstrainedFSCIHGR
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
                test_class_dataloader, desc="test incremental class"
            ):
                label = label.to(self.RANK)
                y = self.ddp_model(image.to(self.RANK))
                y = torch.softmax(y, dim=-1)

                all_logit.append(y)
                all_label.append(label)

        all_logit = torch.cat(all_logit)
        all_label = torch.cat(all_label)

        pred = torch.argmax(all_logit, dim=1)
        acc = (all_label == pred).float().mean() * 100

        print(session)
        if session == 9:
            data_writer.collect(all_label, pred)
            data_writer.write()

        return acc

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
            self.logger.ddp_info(f"Train Epoch: {epoch}")
            self.model.train(True)
            self.logger.ddp_info(f"Current LR: {self.scheduler.get_last_lr()[0]}")
            for epi, (image, noise_image, label, pose, angle) in enumerate(self.dataloader):
                with torch.autocast(device_type="cuda", dtype=self.cfg.TRAIN.DTYPE):
                    image = image.to(self.RANK)
                    label = label.to(self.RANK)

                    y = self.ddp_model(image.to(self.RANK))
                    y = torch.softmax(y, dim=-1)
                    loss_final = torch.nn.functional.cross_entropy(y, label)
                    loss_info = f'Loss: {loss_final.item()}'

                loss_final = loss_final / self.cfg.OPTIM.GRAD_ACCM
                self.scaler.scale(loss_final).backward()

                if epi % self.cfg.OPTIM.GRAD_ACCM == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                torch.cuda.synchronize(device=self.RANK)

                all_time, accm_time, _ = loop_time_counter.step()
                if self.RANK == 0 and epi % self.cfg.LOG.PRINT_FREQ == 0:
                    acc = count_acc(y, label)

                    info = (
                        f"[{epoch:02d}/{self.cfg.TRAIN.EPOCH}][{epi:03d}/{self.cfg.TRAIN.EPISODE}]"
                        + f" | time: epo[{accm_time[0]/3600:.2f}/{all_time[0]/3600:.2f}h] epi[{accm_time[1]/60:.1f}/{all_time[1]/60:.1f}m] | "
                        + loss_info
                        + f" | acc: {acc:.2f}"
                    )

                    self.logger.ddp_info(info)

            self.scheduler.step()

            self.model.train(False)

            if self.RANK == 0:
                if self.cfg.TRAIN.VALIDATION:
                    self.logger.ddp_info(f"incremental class")
                    # class_acc_dt = self.incr_class()
                    class_acc_dt = self.test_class(0)
                    self.logger.ddp_info(pformat(class_acc_dt))

                    if class_acc_dt[0] > best_acc:
                        self.logger.ddp_info(f"Find better model.")
                        best_acc = class_acc_dt[0]

                        os.system(f"rm {self.cfg.LOG.DIR}" + "/ckpt-best-*.pth") if epoch > 0 else None
                        torch.save(
                            self.model.state_dict(),
                            self.cfg.LOG.DIR + f"/ckpt-best-{epoch}-{best_acc}.pth",
                        )
                os.system(f"rm {self.cfg.LOG.DIR}" + "/ckpt-last-*.pth") if epoch > 0 else None
                torch.save(self.model.state_dict(), self.cfg.LOG.DIR + f"/ckpt-last-{epoch}.pth")

            if self.cfg.TRAIN.SAVE_EACH:
                torch.save(self.model.state_dict(), self.cfg.LOG.DIR + f"/ckpt-{epoch}.pth")


    def run(self):
        self.setup_logger()
        # self.setup_model()
        self.model.train(False)

        if "FSCI" in self.cfg.TRAIN.TASK:
            class_acc_dt = self.incr_class()
            self.logger.ddp_info(pformat(class_acc_dt))

        if "FSSI" in self.cfg.TRAIN.TASK:
            shape_acc_dt = self.incr_shape()
            self.logger.ddp_info(pformat(shape_acc_dt))


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

    def ddp_worker1(self, rank=0):
        self.RANK = rank
        self.DEVICE = rank
        self.setup_seed(self.SEED + self.RANK)
        self.setup_logger()
        self.setup_ddp_env(rank, port=str(self.cfg.DDP.MASTER_PORT))
        # self.setup_ddp_dataset()
        self.setup_model()
        self.setup_optimizer()
        self.run()
        self.clean()


class ConfMatData:
    def __init__(self, path) -> None:
        self.pd_ls = []
        self.gt_ls = []
        self.path = path

    def collect(self, gt_ts, pd_ts):
        # lg_ts: logit tensor
        # gt_ts: groundtruth tensor
        self.pd_ls.append(pd_ts.clone().detach().cpu())
        self.gt_ls.append(gt_ts.clone().detach().cpu())

    def write(self):
        import os
        import torch

        pd_ts = torch.cat(self.pd_ls)
        gt_ts = torch.cat(self.gt_ls)
        print('pd.shape', pd_ts.shape)
        print('gt.shape', gt_ts.shape)

        os.makedirs(self.path, exist_ok=True)
        save_path = os.path.join(self.path, 'confuse_matrix_data.pth')
        # torch.save([gt_ts, pd_ts], save_path)

data_writer = ConfMatData('out/confuse_matrix_data/Finetune')