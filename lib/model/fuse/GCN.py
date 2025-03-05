import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_print import tpr
from .sem_graph_conv import (
    SemGraphConv,
    _ResGraphConv,
    _GraphConv,
)
from ...tool.hand_fn import (
    MANO_KPT_TO_ANGLE_ID2,
    MAN0_KPT_LINK,
    MAN0_ANGLE_LINK,
    angle_to_kpt,
    kpt_to_angle,
    MANO_KID,
)


class HandEmbNet(nn.Module):
    def __init__(self, in_dim=768, hid_dim=128, p_dropout=None):
        super(HandEmbNet, self).__init__()

        link_kpt = torch.tensor(MAN0_KPT_LINK)
        adj_kpt = torch.eye(21)
        adj_kpt[link_kpt[:, 0], link_kpt[:, 1]] = 1
        adj_kpt = adj_kpt / adj_kpt.sum(-1, keepdim=True)
        self.register_buffer("adj_kpt", adj_kpt)

        link_ang = torch.tensor(MAN0_ANGLE_LINK)
        adj_ang = torch.eye(15)
        adj_ang[link_ang[:, 0], link_ang[:, 1]] = 1
        adj_ang = adj_ang / adj_ang.sum(-1, keepdim=True)
        self.register_buffer("adj_ang", adj_ang)

        self.block0 = SemGraphConv(in_dim, hid_dim, self.adj_kpt, bias=False)
        self.block1 = nn.Sequential(
            _ResGraphConv(self.adj_kpt, hid_dim, hid_dim, hid_dim, p_dropout),
            _GraphConv(self.adj_kpt, hid_dim, hid_dim, p_dropout),
            _ResGraphConv(self.adj_kpt, hid_dim, hid_dim, hid_dim, p_dropout),
            _GraphConv(self.adj_kpt, hid_dim, hid_dim, p_dropout),
        )

        self.dec_kpt = nn.Sequential(
            SemGraphConv(hid_dim, 3, self.adj_kpt, bias=False),
        )

        self.block2 = nn.Sequential(
            _GraphConv(self.adj_ang, hid_dim, hid_dim, p_dropout),
            _ResGraphConv(self.adj_ang, hid_dim, hid_dim, hid_dim, p_dropout),
            _GraphConv(self.adj_ang, hid_dim, hid_dim, p_dropout),
            SemGraphConv(hid_dim, hid_dim, self.adj_ang, bias=False),
        )

        self.dec_ang = nn.Sequential(
            # _GraphConv(self.adj_ang, hid_dim, hid_dim, p_dropout),
            SemGraphConv(hid_dim, 4, self.adj_ang, bias=False),
        )

        self.dec_emb = nn.Identity()

        KPT_TO_ANG = torch.tensor(MANO_KPT_TO_ANGLE_ID2, dtype=torch.int)
        self.register_buffer("KPT_TO_ANG", KPT_TO_ANG)

        ANG2KPT_VALID = torch.tensor(
            [
                MANO_KID["IP"],
                MANO_KID["ID"],
                MANO_KID["IT"],
                MANO_KID["MP"],
                MANO_KID["MD"],
                MANO_KID["MT"],
                MANO_KID["PP"],
                MANO_KID["PD"],
                MANO_KID["PT"],
                MANO_KID["RP"],
                MANO_KID["RD"],
                MANO_KID["RT"],
                MANO_KID["TP"],
                MANO_KID["TD"],
                MANO_KID["TT"],
            ],
            dtype=torch.int,
        )
        self.register_buffer("ANG2KPT_VALID", ANG2KPT_VALID)

    def forward(self, x):
        # x : (B, 21, 128)
        x = self.block0(x)
        x = self.block1(x)

        pd_kpt = self.dec_kpt(x)
        pd_kpt = pd_kpt.clone()
        pd_kpt[:, :, :2] = pd_kpt[:, :, :2] * 112 + 111.5
        pd_kpt[:, :, 2] = pd_kpt[:, :, 2] * 150

        x = x[:, self.KPT_TO_ANG]  # (B, 15, 128)
        x = self.block2(x)
        pd_ang = self.dec_ang(x)
        pd_emb = self.dec_emb(x)  # (B, 15, 128)
        pd_emb = pd_emb.reshape(pd_emb.shape[0], -1)
        return pd_emb, pd_kpt, pd_ang

    def criterion_kpt(self, pd, gt):
        # gt: (B, 21, 3)
        m = gt[:, 0, -1] < 1000
        if m.sum() > 0:
            pd_kp_ = pd[m]
            gt_kp_ = gt[m]

            pd_kp_[:, :, :2] = (pd_kp_[:, :, :2] - 111.5) / 112
            gt_kp_[:, :, :2] = (gt_kp_[:, :, :2] - 111.5) / 112
            pd_kp_[:, :, 2] = pd_kp_[:, :, 2] / 150
            gt_kp_[:, :, 2] = gt_kp_[:, :, 2] / 150

            pd_kp_ = pd_kp_ * 30
            gt_kp_ = gt_kp_ * 30

            loss = F.smooth_l1_loss(pd_kp_, gt_kp_)
        else:
            loss = 0
        return loss

    def criterion_ang(self, pd, gt, pd_uvd):
        # gt: (B, 15, 4)
        m = gt[:, 0, -1] < 1000

        _pd_uvd = pd_uvd.detach().clone()
        _pd_uvd[:, :, :2] = _pd_uvd[:, :, :2] * 150 / 112
        gt = gt.clone()
        gt[~m] = kpt_to_angle(pd_uvd[~m], wrist_rotation=False).to(gt).detach()

        pd_kpt = angle_to_kpt(pd)
        gt_kpt = angle_to_kpt(gt)

        if m.sum() > 0:
            pd_kpt_ = pd_kpt * 30 / 150
            gt_kpt_ = gt_kpt * 30 / 150

            pd_kpt_ = pd_kpt_[:, self.ANG2KPT_VALID]
            gt_kpt_ = gt_kpt_[:, self.ANG2KPT_VALID]

            loss1 = F.smooth_l1_loss(pd_kpt_[m], gt_kpt_[m])
            loss2 = F.smooth_l1_loss(pd_kpt_[~m], gt_kpt_[~m])

            # loss1 = torch.nan_to_num(loss1)
            # loss2 = torch.nan_to_num(loss2)

            loss = loss1 + loss2 * 0.1

            return loss, (pd_kpt, gt_kpt)
        else:
            return 0, (pd_kpt, gt_kpt)

    def criterion_kpt_sync(self, pd, fake_gt, gt):
        m = gt[:, 0, -1] > 1000
        if m.sum() > 0:
            pd_kp_ = pd[m]
            gt_kp_ = fake_gt[m].detach()

            pd_kp_[:, :, :2] = (pd_kp_[:, :, :2] - 111.5) / 112
            gt_kp_[:, :, :2] = (gt_kp_[:, :, :2] - 111.5) / 112
            pd_kp_[:, :, 2] = pd_kp_[:, :, 2] / 150
            gt_kp_[:, :, 2] = gt_kp_[:, :, 2] / 150

            pd_kp_ = pd_kp_ * 30
            gt_kp_ = gt_kp_ * 30

            loss = F.smooth_l1_loss(gt_kp_, pd_kp_)
            loss = loss
        else:
            loss = 0
        return loss
