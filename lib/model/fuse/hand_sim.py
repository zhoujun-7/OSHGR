import time
from tensor_print import tpr
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sem_graph_conv import (
    SemGraphConv,
    _ResGraphConv,
    _GraphConv,
)


class HandSimNet(nn.Module):
    def __init__(
        self,
        adj_hands,
        adj_ang,
        hid_dim=128,
        num_J=15,
        p_dropout=None,
    ):
        super(HandSimNet, self).__init__()

        self.compare1 = nn.Sequential(
            _ResGraphConv(adj_hands, hid_dim, hid_dim, hid_dim, p_dropout),
            _GraphConv(adj_hands, hid_dim, hid_dim, p_dropout),
            _ResGraphConv(adj_hands, hid_dim, hid_dim, hid_dim, p_dropout),
        )
        self.compare2 = nn.Sequential(
            _GraphConv(adj_ang, hid_dim, hid_dim, p_dropout),
            _ResGraphConv(adj_ang, hid_dim, hid_dim, hid_dim, p_dropout),
            _GraphConv(adj_ang, hid_dim, hid_dim, p_dropout),
        )

        self.dec_quat = nn.Sequential(
            _GraphConv(adj_ang, hid_dim, hid_dim, p_dropout),
            SemGraphConv(hid_dim, 4, adj_ang, bias=False),
        )

        self.dec_att = nn.Sequential(
            # _GraphConv(adj_ang, hid_dim, hid_dim, p_dropout),
            SemGraphConv(hid_dim, hid_dim, adj_ang, bias=False),
        )

    def forward(self, x):
        # x: (B, 2x15, 128)
        x = self.compare1(x)
        x = x.view(-1, 2, 15, 128)
        x = x.swapaxes(1, 2)
        x = x.mean(2)

        x = self.compare2(x)  # (B, 15, 128)

        quat = self.dec_quat(x)  # (B, 15, 4)
        quat = quat / torch.norm(quat, p=2, dim=-1, keepdim=True)

        att = self.dec_att(x)  # (B, 15, 128)
        att = att.sum(-1)
        att = torch.nn.functional.softmax(att, dim=-1)  # (B, 15)
        # att = F.normalize(att, p=2, dim=-1)
        att = att * att.shape[1]

        return quat, att
