from __future__ import absolute_import, division

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# class SemGraphConv(nn.Module):
#     """
#     Semantic graph convolution layer
#     """

#     def __init__(self, in_features, out_features, adj, bias=True):
#         super(SemGraphConv, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)

#         self.adj = adj
#         self.m = self.adj > 0
#         self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
#         nn.init.constant_(self.e.data, 1)

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
#             stdv = 1.0 / math.sqrt(self.W.size(2))
#             self.bias.data.uniform_(-stdv, stdv)
#         else:
#             self.register_parameter("bias", None)

#         adj = -9e15 * torch.ones_like(self.adj)
#         adj[self.m] = self.e
#         adj = F.softmax(adj, dim=1)
#         M = torch.eye(adj.size(0), dtype=torch.float)

#         self.register_buffer("adj_", adj)
#         self.register_buffer("M_", M)

#     def forward(self, input):
#         h0 = torch.matmul(input, self.W[0])
#         h1 = torch.matmul(input, self.W[1])
#         output = torch.matmul(self.adj_ * self.M_, h0) + torch.matmul(self.adj_ * (1 - self.M_), h1)

#         if self.bias is not None:
#             return output + self.bias.view(1, 1, -1)
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = self.adj > 0
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1.0 / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj, device=input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float, device=input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out
