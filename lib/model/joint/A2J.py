import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Anchor2Joint(nn.Module):
    def __init__(self, is_detach=False) -> None:
        super(Anchor2Joint, self).__init__()
        self.is_detach = is_detach

        X = np.arange(0, 224, 4)
        Y = np.arange(0, 224, 4)
        XX, YY = np.meshgrid(X, Y)
        XX = XX[None]
        YY = YY[None]
        anchor = np.concatenate([XX, YY], axis=0)  # (2, 56, 56)
        anchor = torch.from_numpy(anchor)
        self.register_buffer("anchor", anchor.to(torch.float32))

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=768,
                out_channels=256,
                kernel_size=(4),
                stride=2,
                padding=1,
            ),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(4),
                stride=2,
                padding=1,
            ),
        )
        self.decoder_UV = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=21,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.decoder_D = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=21,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x = self.deconv(x)  # (B, 128, 56, 56)
        x_uv = self.decoder_UV(x)  # (B, 21, 56, 56)
        x_d = self.decoder_D(x)  # (B, 21, 56, 56)

        x_uv_ = x_uv.clone()

        B, J, H, W = x_uv.shape
        att_map = x_uv.clone()
        att_map = F.interpolate(att_map, (14, 14), mode="bilinear")
        att_map = F.softmax(att_map.view(B, J, -1), dim=-1).view(B, J, 14, 14)
        if self.is_detach:
            att_map = att_map.detach()

        x_uv = F.softmax(x_uv.view(B, J, -1), dim=-1).view(B, J, -1, H, W)
        x_d = x_d.view(B, J, 1, H, W)

        uv_map = x_uv * self.anchor
        d_map = x_uv * x_d

        uv = uv_map.sum(-1).sum(-1)
        d = d_map.sum(-1).sum(-1)
        uvd = torch.cat([uv, d], dim=2)  # (B, J, 3)

        return uvd, att_map, x_uv_

    def criterion_uvd(self, pd, gt):
        m = gt[:, 0, -1] < 1000
        if m.sum() > 0:
            gt_kp_ = gt[m]
            pd_kp_ = pd[m]

            pd_kp_[:, :, :2] = (pd_kp_[:, :, :2] - 111.5) / 112
            gt_kp_[:, :, :2] = (gt_kp_[:, :, :2] - 111.5) / 112
            pd_kp_[:, :, 2] = pd_kp_[:, :, 2] / 150
            gt_kp_[:, :, 2] = gt_kp_[:, :, 2] / 150

            pd_kp_ = pd_kp_ * 30
            gt_kp_ = gt_kp_ * 30

            loss = F.smooth_l1_loss(gt_kp_, pd_kp_)
        else:
            loss = 0
        return loss
