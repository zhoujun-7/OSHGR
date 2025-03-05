import torch
import time
import tqdm
import math
import timm
import torch.nn as nn
from tensor_print import tpr
import torch.nn.functional as F

from .backbone.resnet import ResNet18
from .backbone.vit import VisionTransformer
from .classifier.cos import CosClassifier
from .semantic.MAE import MaskAutoEncoder
from .joint.A2J import Anchor2Joint
from .fuse.GCN import HandEmbNet

from ..tool.hand_fn import (
    MANO_KPT_TO_ANGLE_ID2,
    MAN0_KPT_LINK,
    MAN0_ANGLE_LINK,
    angle_to_kpt,
    kpt_to_angle,
    MANO_KID,
)
from ..tool.visulize_fn import detach_and_draw_batch_img, draw_image_pose, save_dealed_batch


class BaseNet(nn.Module):
    def __init__(self, EMB_DIM=512, CLS_NUM=68, BASE_CLASS=23, **kwargs):
        super(BaseNet, self).__init__()

        self.encoder = ResNet18()
        self.classifier = CosClassifier(emb_dim=EMB_DIM, cls_num=CLS_NUM, base_class=BASE_CLASS)
        self.cls_num = CLS_NUM

    def forward(self, x={}, gt={}, phase="pretrain"):
        if phase == "pretrain":
            pd_emb = self.encoder(x["noise"])
            pd_logit = self.classifier(pd_emb)
            pd = {
                "emb": pd_emb,
                "logit": pd_logit,
            }

            loss_class = self.classifier.criterion(pd["logit"], gt["class"])
            loss = {"logit": loss_class}

            return pd, loss

        elif phase == "test":
            pd_emb = self.encoder(x["image"])
            pd_logit = self.classifier(pd_emb)

            pd = {
                "emb": pd_emb,
                "logit": pd_logit,
            }

            loss = {}

            return pd, loss

    def incremental_training(self, dataloader, num_augment=1, *args, **kwargs):
        with torch.no_grad():
            embedding_list = []
            label_list = []
            device = self.classifier.proto.weight.data.device

            desc = str(dataloader.dataset).split(" object")[0].split(".")[-1] + f": {len(dataloader.dataset)}"
            for i in range(num_augment):
                for data in tqdm.tqdm(dataloader, desc=desc):
                    x = {"image": data[0].to(device)}
                    gt = {"class": data[2].to(device)}
                    pd, loss = self.forward(x, gt, phase="test")
                    embedding_list.append(pd["emb"].cpu())
                    label_list.append(gt["class"].cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            new_proto = self.classifier.proto.weight.data.clone()
            for class_index in label_list.unique():
                data_index = label_list == class_index
                embedding_this = embedding_list[data_index]
                embedding_this = embedding_this.mean(0)
                new_proto[class_index] = embedding_this

            self.classifier.proto.weight.data = new_proto

    def average_prototype(self, *args, **kwargs):
        self.incremental_training(*args, **kwargs)


class MyNet(nn.Module):
    def __init__(
        self,
        ONLY_MAE=False,
        MAE_USE=True,
        MAE_FROM_NOISE=True,
        A2J_USE=True,
        A2J_LOC=7,
        A2J_DETACH=False,
        DOWNSAMPLE="avg_pool",
        GCN_USE=True,
        GCN_IN_DIM=768,
        GCN_HID_DIM=128,
        CLS_NUM=68,
        NUM_BASE_CLASS=23,
        **kwargs,
    ):
        super(MyNet, self).__init__()

        if MAE_USE:
            self.decoder_MAE = MaskAutoEncoder()
            self.backbone = self.decoder_MAE.forward_VIT
        else:
            self.backbone = VisionTransformer()

        if not ONLY_MAE:
            if A2J_USE:
                self.decoder_A2J = Anchor2Joint(A2J_DETACH)

            if GCN_USE:
                self.decoder_GCN = HandEmbNet(GCN_IN_DIM, GCN_HID_DIM)
                emb_dim = 15 * GCN_HID_DIM
            else:
                emb_dim = GCN_IN_DIM
        else:
            emb_dim = GCN_IN_DIM

        self.classifier = CosClassifier(emb_dim=emb_dim, cls_num=CLS_NUM, base_class=NUM_BASE_CLASS)
        self.cls_num = CLS_NUM

        self.ONLY_MAE = ONLY_MAE
        self.MAE_USE = MAE_USE
        self.MAE_FROM_NOISE = MAE_FROM_NOISE
        self.A2J_USE = A2J_USE
        self.A2J_LOC = A2J_LOC
        self.GCN_USE = GCN_USE
        self.DOWNSAMPLE = DOWNSAMPLE

    def unpatchify(self, x):
        B, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, -1, 14, 14).contiguous()
        return x

    def forward(self, x={}, gt={}, phase="pretrain"):
        pd = {}
        loss = {}
        if phase == "pretrain":
            if self.MAE_USE:
                if self.MAE_FROM_NOISE:
                    loss_mae, pd_mae, mask = self.decoder_MAE(x["image"], x["noise"])
                else:
                    loss_mae, pd_mae, mask = self.decoder_MAE(x["image"], x["image"])

                pd["mae"] = pd_mae
                loss["mae"] = loss_mae

            if not self.ONLY_MAE:
                feat, feat_ls = self.backbone(x["noise"])
                if self.A2J_USE:
                    feat_a2j = self.unpatchify(feat_ls[self.A2J_LOC][:, 1:])
                    pd_uvd, att_map, att_map_ = self.decoder_A2J(feat_a2j)
                    B, J, _, _ = att_map.shape
                    att_map = att_map.reshape(B, J, -1)  # (B, J, 196)
                    pd["uvd"] = pd_uvd

                    pd["att_map"] = att_map_

                    loss_uvd = self.decoder_A2J.criterion_uvd(pd["uvd"], gt["uvd"])
                    loss["uvd"] = loss_uvd

                if self.DOWNSAMPLE == "avg_pool":
                    feat_gcn = feat_ls[-1][:, 1:].mean(1)
                elif self.DOWNSAMPLE == "spatial":
                    assert self.A2J_USE
                    feat_gcn = att_map @ feat_ls[-1][:, 1:]

                if self.GCN_USE:
                    pd_emb, pd_kpt, pd_ang = self.decoder_GCN(feat_gcn)
                    pd["emb"] = pd_emb
                    pd["kpt"] = pd_kpt
                    pd["ang"] = pd_ang

                    loss_kpt = self.decoder_GCN.criterion_kpt(pd["kpt"], gt["uvd"])
                    loss_ang, pd_ang2kpt = self.decoder_GCN.criterion_ang(pd["ang"], gt["ang"], pd["uvd"])
                    loss_syn = self.decoder_GCN.criterion_kpt_sync(pd["kpt"], pd["uvd"], gt["uvd"])
                    loss["kpt"] = loss_kpt
                    loss["ang"] = loss_ang
                    loss["syn"] = loss_syn

                    pd["ang2kpt"] = pd_ang2kpt
                else:
                    pd["emb"] = feat_gcn

                pd_logit = self.classifier(pd["emb"])
                loss_logit = self.classifier.criterion(pd_logit, gt["class"])
                pd["logit"] = pd_logit
                loss["logit"] = loss_logit
            else:
                pd["logit"] = 0
                loss["logit"] = 0

            return pd, loss

        elif phase == "test":
            feat, feat_ls = self.backbone(x["image"])

            if self.A2J_USE:
                feat_a2j = self.unpatchify(feat_ls[self.A2J_LOC][:, 1:])
                pd_uvd, att_map, att_map_ = self.decoder_A2J(feat_a2j)
                B, J, _, _ = att_map.shape
                att_map = att_map.reshape(B, J, -1)  # (B, J, 196)
                pd["uvd"] = pd_uvd
                pd['att_map'] = att_map_

            if self.DOWNSAMPLE == "avg_pool":
                feat_gcn = feat_ls[-1][:, 1:].mean(1)
            elif self.DOWNSAMPLE == "spatial":
                assert self.A2J_USE
                feat_gcn = att_map @ feat_ls[-1][:, 1:]

            if self.GCN_USE:
                pd_emb, pd_kpt, pd_ang = self.decoder_GCN(feat_gcn)
                pd["emb"] = pd_emb
                pd["kpt"] = pd_kpt
                pd["ang"] = pd_ang
                pd["ang2kpt"] = angle_to_kpt(pd["ang"])
            else:
                pd["emb"] = feat_gcn

            pd_logit = self.classifier(pd["emb"])
            pd["logit"] = pd_logit

            return pd, loss

    def incremental_training(self, dataloader, num_augment=1, *args, **kwargs):
        with torch.no_grad():
            embedding_list = []
            label_list = []
            device = self.classifier.proto.weight.data.device

            desc = str(dataloader.dataset).split(" object")[0].split(".")[-1] + f": {len(dataloader.dataset)}"
            for i in range(num_augment):
                for data in tqdm.tqdm(dataloader, desc=desc):
                    x = {"image": data[0].to(device)}
                    gt = {"class": data[2].to(device)}

                    pd, loss = self.forward(x, gt, phase="test")

                    embedding_list.append(pd["emb"].cpu())
                    label_list.append(gt["class"].cpu())
            embedding_list = torch.cat(embedding_list, dim=0)

            label_list = torch.cat(label_list, dim=0)

            new_proto = self.classifier.proto.weight.data.clone()

            num_new_proto = label_list.unique().max() + 1
            if num_new_proto > new_proto.shape[0]:
                append_proto = torch.zeros([num_new_proto - new_proto.shape[0], new_proto.shape[1]]).to(new_proto)
                nn.init.kaiming_uniform_(append_proto, a=math.sqrt(5))
                new_proto = torch.cat([new_proto, append_proto])

            for class_index in label_list.unique():
                data_index = label_list == class_index
                embedding_this = embedding_list[data_index]
                embedding_this = embedding_this.mean(0)
                new_proto[class_index] = embedding_this

            self.classifier.proto.weight.data = new_proto

    def average_prototype(self, *args, **kwargs):
        self.incremental_training(*args, **kwargs)
