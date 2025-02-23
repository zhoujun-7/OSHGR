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
from ..dataset.oshgr import (
    UnconstrainedPretrain,
    UnconstrainedFSSIHGR,
    UnconstrainedFSCIHGR,
    ConstrainedFSSIHGR,
    ConstrainedFSCIHGR,
    ViewPointRPY,
)
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
            for n, p in ckpt.items():
                if "classifier" in n:
                    continue
                self.model.state_dict()[n].copy_(p)
            # self.model.load_state_dict(ckpt)
        else:
            self.logger.ddp_info("No checkpoint is specified.")

        self.model.to(self.RANK)

    def incr_shape(self):
        if self.cfg.TRAIN.TASK == "CFSSI":
            FSSIHGR = ConstrainedFSSIHGR
        elif self.cfg.TRAIN.TASK == "UFSSI":
            FSSIHGR = UnconstrainedFSSIHGR

        shape_ls = ["ZhouJun", "Glove_Thin", "Glove_Thick", "Glove_Half-finger"]
        shape_acc_dt = {}
        shape_test_ls = []
        for i, shape in enumerate(shape_ls):
            incr_shape_dataset = FSSIHGR(
                PHASE="incremental",
                INCR_SHAPE=shape,
                INCR_SHOT=self.cfg.DATASET.INCR_SHOT,
                IS_AUGMENT=self.cfg.AUGMENT.INCR_AUGMENT,
            )
            incr_shape_dataset.label_ls += self.cfg.DATASET.NUM_BASE_CLASS * (i + 1)

            print('incr label', np.unique(incr_shape_dataset.label_ls))

            incr_shape_dataloader = DataLoader(
                dataset=incr_shape_dataset,
                num_workers=0,
                batch_size=64,
                shuffle=False,
            )

            # self.model.classifier.restore_proto(is_kaiming=True)
            self.model.incremental_training(incr_shape_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)
            print(self.model.classifier.proto.weight.data.shape)
            shape_test_ls.append(shape)

            if len(shape_test_ls) == 4:
                acc = self.test_shape(shape_test_ls)
                shape_acc_dt[shape] = acc

        return shape_acc_dt

    def test_shape(self, shape_ls):
        if self.cfg.TRAIN.TASK == "CFSSI":
            FSSIHGR = ConstrainedFSSIHGR
        elif self.cfg.TRAIN.TASK == "UFSSI":
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

        # test_shape_dataset = FSSIHGR(
        #     PHASE="test",
        #     INCR_SHAPE=shape_ls[-1],
        # )

        test_shape_dataloader = DataLoader(
            dataset=test_shape_dataset,
            num_workers=0,
            pin_memory=True,
            batch_size=64,
            shuffle=False,
        )

        print('dataset length', len(test_shape_dataloader))

        all_logit = []
        all_label = []
        with torch.no_grad():
            for image, noise_image, label, pose, angle in tqdm.tqdm(
                test_shape_dataloader, desc=f"test shape {shape_ls[-1]} {len(test_shape_dataset)}"
            ):
                x_dt = {"image": image.to(self.RANK)}
                gt_dt = {"class": label.to(self.RANK)}
                pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test")
                all_logit.append(pd_dt["logit"])
                all_label.append(gt_dt["class"])

                if len(dataset_ls) == 4:
                    vizer.is_shape = True
                    # vizer.write(image, gt_dt['class'], pd_dt["logit"], pd_dt["kpt"], angles_ts=pd_dt["ang2kpt"])
                    vizer.write(image, gt_dt['class'], pd_dt["logit"])

        all_logit = torch.cat(all_logit)
        all_label = torch.cat(all_label)

        all_pred = torch.argmax(all_logit, dim=1)
        all_pred = all_pred % self.cfg.DATASET.NUM_BASE_CLASS
        all_label = all_label % self.cfg.DATASET.NUM_BASE_CLASS
        # print(all_label[-100:])
        # print(all_pred[-100:])

        acc = (all_pred == all_label).to(torch.float32).mean().item() * 100
        acc = float(f"{acc:.2f}")

        # print(all_label[::500])
        # print(all_pred[::500])
        # print(all_label != all_pred)
        # print(acc)

        # print(all_logit[all_label != all_pred])
        # exit()

        return acc

    def incr_class(self):
        if self.cfg.TRAIN.TASK == "CFSCI":
            FSCIHGR = ConstrainedFSCIHGR
        elif self.cfg.TRAIN.TASK == "UFSCI":
            FSCIHGR = UnconstrainedFSCIHGR

        self.model.classifier.restore_proto(is_kaiming=True)
        for sess in range(1, self.cfg.DATASET.NUM_SESSION + 1):
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
            self.model.incremental_training(incr_class_dataloader, self.cfg.AUGMENT.NUM_INCR_AUGMENT)

        class_acc_dt = self.test_class(sess)

        return class_acc_dt

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

                vizer.write(image, gt_dt['class'], pd_dt["logit"], pd_dt["kpt"], angles_ts=pd_dt["ang2kpt"], att_map=pd_dt["att_map"])
                # vizer.write(image, gt_dt['class'], pd_dt["logit"])

        all_logit = torch.cat(all_logit)
        all_label = torch.cat(all_label)

        od1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        od = [0 for i in range(68)]
        for i in range(68):
            od[od1[i]] = i
        od = torch.tensor(od).cuda()

        class_acc_dt = {}
        for sess in range(session + 1):
            num_test_class = self.cfg.DATASET.NUM_BASE_CLASS + self.cfg.DATASET.INCR_WAY * sess
            # m = od[all_label] < num_test_class
            m = all_label < num_test_class
            sess_label = all_label[m]
            sess_pred = torch.argmax(all_logit[m][:, :num_test_class], dim=1)

            sess_acc = (sess_pred == sess_label).to(torch.float32).mean().item() * 100
            sess_acc = float(f"{sess_acc:.2f}")
            class_acc_dt[sess] = sess_acc

            mask = sess_pred != sess_label
            print('*'*100)
            print(sess_pred[mask])
            print(sess_label[mask])
        return class_acc_dt
    
    def infer_pose(self):
        infer_yaw_dataset = ViewPointRPY("Left_to_Right")
        infer_pitch_dataset = ViewPointRPY("Up_to_Down")
        infer_dataset = torch.utils.data.ConcatDataset([infer_yaw_dataset, infer_pitch_dataset])
        infer_dataloader = DataLoader(
            dataset=infer_dataset,
            num_workers=0,
            batch_size=64,
            shuffle=False,
        )
        with torch.no_grad():
            for image, noise_image, label, view_point in tqdm.tqdm(
                infer_dataloader, desc=f"infer pose {len(infer_dataset)}"
            ):
                x_dt = {"image": image.to(self.RANK)}
                gt_dt = {"class": label.to(self.RANK)}
                pd_dt, loss_dt = self.model(x_dt, gt_dt, phase="test")
                vizer.write(image, gt_dt['class'], pd_dt["logit"], pd_dt["kpt"], angles_ts=pd_dt["ang2kpt"])

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

        # if "FSCI" in self.cfg.TRAIN.TASK:
        #     class_acc_dt = self.incr_class()
        #     self.logger.ddp_info(pformat(class_acc_dt))

        # if "FSSI" in self.cfg.TRAIN.TASK:
        #     shape_acc_dt = self.incr_shape()
        #     self.logger.ddp_info(pformat(shape_acc_dt))

        self.infer_pose()

import os
import torch
import cv2
import numpy as np

class WriteVizResult:

    def __init__(self, path, step=10):
        self.path = path
        self.num = 0
        self.step = step
        self.is_shape = False

    def write(self, imgs_ts, labels_ts, logits_ts, joints_ts=None, angles_ts=None, att_map=None):
        B, _, H, W = imgs_ts.shape
        if angles_ts is not None:
            angles_ts = self.local_pose_2_image_pose(angles_ts)
        for b_i in range(B):
            self.num += 1
            if self.num % self.step == 0:
                if joints_ts is None:
                    j_i = None
                else:
                    j_i = joints_ts[b_i]

                if angles_ts is None:
                    a_i = None
                else:
                    a_i = angles_ts[b_i]

                if att_map is None:
                    att_i = None
                else:
                    att_i = att_map[b_i]
                self._write(imgs_ts[b_i], labels_ts[b_i], logits_ts[b_i], j_i, a_i, att_i)
                
            
    def _write(self, img_ts, label_ts, logits_ts, joint_ts=None, angle_ts=None, att_map_ts=None):
        img_np = img_ts.clone().detach().cpu().numpy()
        label_np = label_ts.clone().detach().cpu().numpy()
        logits_np = logits_ts.clone().detach().cpu().numpy()
        
        rgb_np = self.depth_to_rgb(img_np[0], d_min=-1, d_max=1)
        # logits_per_np = logits_np / logits_np.max()
        logits_per_np = logits_np

        if self.is_shape:
            logits_per_np = logits_per_np.reshape(-1, 5, 23)
            logits_per_np = np.max(logits_per_np, axis=-2)
            logits_per_np = np.max(logits_per_np, axis=0)
            num_class = 23
        else:
            num_class = 68

        rank = np.argsort(logits_per_np)[::-1]
        save_name = f'id-{self.num}_label-{label_np}.png' 
        txt_str = 'rank'
        for i in range(num_class):
            txt_str += f'{rank[i]}_{logits_per_np[rank[i]]:.2f}, '
            
        path = os.path.join(self.path, 'rgb', save_name)
        txt_path = os.path.join(self.path, 'txt', f'id-{self.num}_label-{label_np}.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, 'w') as f:
            f.write(txt_str)

        cv2.imwrite(path, rgb_np)

        if joint_ts is not None:
            joint_np = joint_ts.clone().detach().cpu().numpy()
            rgb_joint_np = self.draw_skeleton(rgb_np, joint_np)
            path = os.path.join(self.path, 'joint', save_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, rgb_joint_np)

            rgb_joint_np = self.draw_skeleton(np.ones_like(rgb_np)*255, joint_np)
            path = os.path.join(self.path, 'joint_bg', save_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, rgb_joint_np)
        
        if angle_ts is not None:
            angle_np = angle_ts.clone().detach().cpu().numpy()
            rgb_angle_np = self.draw_angle_pose(rgb_np, angle_np)
            path = os.path.join(self.path, 'angle', save_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, rgb_angle_np)

        if att_map_ts is not None:
            att_map_np = att_map_ts.clone().detach().cpu().numpy()
            save_name = f'id-{self.num}_label-{label_np}.npy' 
            path = os.path.join(self.path, 'att_map', save_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, att_map_np)
        
    @staticmethod
    def depth_to_rgb(image, d_min=None, d_max=None, fake_color=False):
        # bina: (H, W)
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        else:
            image = image.copy()

        H, W = image.shape[0], image.shape[1]
        d_min = image.min() if d_min is None else d_min
        d_max = image.max() if d_max is None else d_max

        rgb = np.zeros([H, W, 3], dtype=np.uint8)
        image = (image - d_min) / (d_max - d_min + 1e-5) * 255
        image = image.astype(np.uint8)

        image = image[..., None]
        rgb = rgb + image

        return rgb
    

    @staticmethod
    def draw_skeleton(img, kpts):
        MANOPTH_ORDER = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [0, 9, 10, 11, 12],
                [0, 13, 14, 15, 16],
                [0, 17, 18, 19, 20],
            ]
        kpt_color = [
                [0, 127, 255],
                [255, 0, 0],
                [255, 165, 0],
                [0, 255, 0],
                [0, 0, 255],
                [139, 0, 255],
            ]
        
        img = img.copy()
        kpts = np.nan_to_num(kpts)

        for i in range(5):
            pt = kpts[MANOPTH_ORDER[i]]
            for j in range(len(pt)):
                src = (int(pt[j][0]), int(pt[j][1]))
                if j < len(pt) - 1:
                    dst = (int(pt[j + 1][0]), int(pt[j + 1][1]))
                    cv2.line(img, src, dst, color=kpt_color[i], thickness=1)
                cv2.circle(img, src, radius=2, color=kpt_color[i], thickness=2)
        # cv2.circle(img, pw, radius=2, color=kpt_color[i], thickness=2)
        return img
    

    def draw_angle_pose(self, rgb, angle):
        dummy_img = np.ones_like(rgb)
        dummy_img *= 255

        return self.draw_skeleton(dummy_img, angle)
    
    @staticmethod
    def local_pose_2_image_pose(local_pose, image_size=(224, 224), k_type="manopth"):
        # local_pose : (B, 21, 3) or (21, 3)

        KID = {  #  mano pytorch, not mano original
            "W_": 0,
            "TM": 1,
            "TP": 2,
            "TD": 3,
            "TT": 4,
            "IM": 5,
            "IP": 6,
            "ID": 7,
            "IT": 8,
            "MM": 9,
            "MP": 10,
            "MD": 11,
            "MT": 12,
            "RM": 13,
            "RP": 14,
            "RD": 15,
            "RT": 16,
            "PM": 17,
            "PP": 18,
            "PD": 19,
            "PT": 20,
        }
        MANO_KID = KID

        local_pose = local_pose.clone().float()

        x_axis = torch.cat(
            [
                local_pose[..., [KID["IM"]], :] - local_pose[..., [KID["MM"]], :],
                local_pose[..., [KID["MM"]], :] - local_pose[..., [KID["RM"]], :],
                local_pose[..., [KID["RM"]], :] - local_pose[..., [KID["PM"]], :],
                local_pose[..., [KID["IM"]], :] - local_pose[..., [KID["RM"]], :],
                local_pose[..., [KID["MM"]], :] - local_pose[..., [KID["PM"]], :],
                local_pose[..., [KID["IM"]], :] - local_pose[..., [KID["PM"]], :],
            ],
            axis=-2,
        )
        x_axis = norm_vec(x_axis)
        x_axis = x_axis.mean(-2)
        x_axis = norm_vec(x_axis)

        y_axis = torch.cat(
            [
                local_pose[..., [KID["W_"]], :] - local_pose[..., [KID["IM"]], :],
                local_pose[..., [KID["W_"]], :] - local_pose[..., [KID["MM"]], :],
                local_pose[..., [KID["W_"]], :] - local_pose[..., [KID["RM"]], :],
            ],
            axis=-2,
        )
        y_axis = norm_vec(y_axis)
        y_axis = y_axis.mean(-2)
        y_axis = norm_vec(y_axis)

        o_mm_P = local_pose[..., MANO_KID["MM"], :]

        z_axis = torch.cross(x_axis, y_axis)
        x_axis = torch.cross(y_axis, z_axis)

        o_mm_T = torch.zeros([local_pose.shape[0], 4, 4], device=local_pose.device, dtype=local_pose.dtype)
        o_mm_T[..., 3, 3] = 1
        o_mm_T[..., :3, 0] = x_axis
        o_mm_T[..., :3, 1] = y_axis
        o_mm_T[..., :3, 2] = z_axis
        o_mm_T[..., :3, 3] = o_mm_P

        mm_o_T = torch.linalg.inv(o_mm_T)
        mm_o_T = torch.nan_to_num(mm_o_T, nan=0.0)

        image_pose = rt_trans(mm_o_T, local_pose)
        image_pose[..., 0] = image_pose[..., 0] + image_size[0] // 2
        image_pose[..., 1] = image_pose[..., 1] + image_size[1] // 2

        return image_pose

def get_vec_length(vec):
    if isinstance(vec, torch.Tensor):
        return torch.sqrt(torch.sum(torch.pow(vec, 2), dim=-1, keepdim=True))
    elif isinstance(vec, np.ndarray):
        return np.sqrt(np.sum(vec**2, axis=-1, keepdims=True))

def norm_vec(vec):
    # vec: (..., 3)
    v = vec / get_vec_length(vec)
    return v

def rt_trans(RT, P):
    """
    Support batch data.
    p.shape = (n, 3) or (B, n, 3)
    """
    assert P.shape[-1] == 3
    if len(P.shape) == 3:
        op = "bni,bji->bnj"
    elif len(P.shape) == 2:
        op = "ni, ji->nj"
    elif len(P.shape) == 4:
        op = "abni,abji->abnj"

    if RT.shape[-1] == 4:
        is_sum = True
    elif RT.shape[-1] == 3:
        is_sum = False

    if isinstance(RT, np.ndarray):
        new_P = np.einsum(op, P, RT[..., :3, :3])
        if is_sum:
            new_P = new_P + RT[..., :3, [3]].swapaxes(-1, -2)
    elif isinstance(RT, torch.Tensor):
        new_P = torch.einsum(op, P, RT[..., :3, :3])
        if is_sum:
            new_P = new_P + RT[..., :3, [3]].swapaxes(-1, -2)
    return new_P

vizer = WriteVizResult('out/viz/rotation_pose_and_joint/class', step=1)