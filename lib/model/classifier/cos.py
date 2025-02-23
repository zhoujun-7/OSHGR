import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_print import tpr
import torch.distributed as dist


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

    def forward(self, emb):
        logit = self.forward_cos(emb, self.proto)
        return logit

    def forward_cos(self, x, proto):
        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(proto.weight, p=2, dim=-1))
        x = 16 * x
        return x

    def criterion(self, logit, tg):
        m = tg < 1000
        if m.sum() > 0:
            loss = F.cross_entropy(logit[m], tg[m])
        else:
            loss = 0
        return loss
