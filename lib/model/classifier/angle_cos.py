import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_print import tpr
import torch.distributed as dist
import numpy as np
import datetime

# get current beijing time in ms
def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')


class CosClassifier(nn.Module):
    def __init__(self, emb_dim=15 * 128, cls_num=68, base_class=23):
        super(CosClassifier, self).__init__()

        self.proto = nn.Linear(emb_dim, cls_num, bias=False)
        nn.init.kaiming_uniform_(self.proto.weight.data, a=math.sqrt(5))

        self.proto_store = None
        self.base_class = list(range(base_class))
        self.incr_class = list(range(base_class, cls_num))

    def save_proto(self):
        self.proto_store = self.proto.weight.data[self.incr_class].clone()

    def restore_proto(self, is_kaiming=True):
        if is_kaiming or self.proto_store is None:
            # self.proto.weight.data[self.incr_class] *= 0
            nn.init.kaiming_uniform_(self.proto.weight.data[23:], a=math.sqrt(5))
        else:
            if not self.proto_store is None:
                # self.proto.weight.data[self.incr_class] = self.proto_store.clone()
                self.proto.weight.data = self.proto_store.clone()

    def forward(self, emb, save=False):
        logit = self.forward_cos(emb, self.proto, save=save)
        return logit
    
    def forward_ori(self, x):
        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.proto.weight, p=2, dim=-1))
        x = 16 * x
        return x

    def forward_cos(self, x, proto, save=False):
        B, _ = x.shape
        N, _ = proto.weight.shape

        _x = x[:, :-15*3]
        _x_ang = x[:, -15*3:].reshape(-1, 1, 15, 3)

        _proto = proto.weight[:, :-15*3]
        _proto_ang = proto.weight[:, -15*3:].reshape(1, -1, 15, 3)

        
        # print(_x_ang[0, 0])
        # print(_proto_ang[0, 0])
        
        ang_w = torch.norm(_x_ang - _proto_ang, dim=-1)  # (B, N, 15)
        # print(ang_w[0, :2])
        # ang_w = F.softmax(-1/ang_w, -1)[..., None]  # (B, N, 15, 1)
        ang_ind_w = torch.argsort(ang_w, dim=-1, descending=True)
        # ang_ind_w = torch.tile(ang_ind_w[..., None], dims=[1, 1, 1, 128])

        # print('*'*100)
        # print(ang_ind_w.shape)
        
        grid_d0, grid_d1, grid_d2 = torch.meshgrid(torch.arange(ang_ind_w.shape[0]), torch.arange(ang_ind_w.shape[1]), torch.arange(ang_ind_w.shape[2]), indexing='ij')

        ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        grid_d0 = grid_d0[:, :,     ind]
        grid_d1 = grid_d1[:, :,     ind]
        ang_ind_w = grid_d2[:, :,   ind]

        # ang_w2 = F.softmax(torch.abs(ang_w - torch.mean(ang_w, dim=-1, keepdim=True)) / 150, dim=-1) * 15
        # ang_w2 = F.softmax(-ang_w / 250, dim=-1) * 15
        ang_w2 = F.softmax(ang_w / 200, dim=-1) * 15

        # print(ang_w2[0, [0, 1]])
        # print(ang_w2[0, 0].sum())
        # exit()
        # print(ang_w2.shape)
        # exit(0)

        # print(grid_d0.shape)
        # print(grid_d1.shape)

        _x = _x.reshape(-1, 1, 15, 128)
        _x = torch.tile(_x, [1, N, 1, 1])
        _x = _x[grid_d0, grid_d1, ang_ind_w]
        _x = F.normalize(_x.reshape(B, N, -1), p=2, dim=-1)
        
        _proto = _proto.reshape(1, -1, 15, 128)
        _proto = torch.tile(_proto, [B, 1, 1, 1])
        _proto = _proto[grid_d0, grid_d1, ang_ind_w]
        _proto = F.normalize(_proto.reshape(B, N, -1), p=2, dim=-1)
        logit = _x * _proto  # (B, N, 15, 128)

        logit = logit.reshape(B, N, 15, 128) * ang_w2[..., None].clone().detach()
        logit = logit.reshape(B, N, -1)
        
        # logit = logit * ang_w
        # logit = logit[grid_d0, grid_d1, ang_ind_w]
        # logit = logit[:, :, :]

        # print(logit[ang_ind_w].shape)
        # print(logit[ang_ind_w].reshape(B, N, 10, -1).shape)
        # exit()

        # region [240116]
        # if save:
        #     _logit = _x * _proto
        #     _logit = _logit.reshape(B, N, 15, 128)
        #     _logit = _logit.detach().cpu().numpy()
        #     _ang_w2 = ang_w2.clone().detach().cpu().numpy()
        #     save_dt = {'logit': _logit, 'ang_w2': _ang_w2}
        #     path = "out/joint-weight/logit_and_weight/" + get_current_time() + ".npz"
        #     np.savez(path, **save_dt)
        # endregion [240116]

        logit = logit.sum(-1) # 
        logit = logit * 16

        # _x = x.reshape(B, 1, 15, 3)
        # _proto = proto.weight.reshape(1, N, 15, 3)
        # x = torch.mean(torch.norm(_x - _proto, dim=- 1), -1)
        # x = -x

        # x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(proto.weight, p=2, dim=-1))
        # x = 16 * x
        return logit

    def criterion(self, logit, tg):
        m = tg < 1000
        if m.sum() > 0:
            loss = F.cross_entropy(logit[m], tg[m])
        else:
            loss = 0
        return loss
