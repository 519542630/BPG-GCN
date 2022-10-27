# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:00:27 2022

@author: nkliu
"""

import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1,
                 ):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res

        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_point, block_size, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations= [1, 2], aug=False):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)
        
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)
        self.aug = aug
        if self.aug:
            self.ca = ChannelAttention(in_planes=out_channels)
            self.sa = SpatialAttention()

    def forward(self, x, keep_prob):
        x1 = self.tcn1(self.gcn1(x)) 
        x1 = self.dropT_skip(self.dropSke(x1, keep_prob, self.A), keep_prob)

        x2 = self.residual(x)
        x2 = self.dropT_skip(self.dropSke(x2, keep_prob, self.A), keep_prob)

        if self.aug:
            x1 = self.ca(x1)
            x1 = self.sa(x1) * x1
            x2 = self.ca(x2)
            x2 = self.sa(x2) * x2

        x = x1 + x2
        x = x.contiguous()
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.fc_2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.fc_3 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.fc_4 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.fc_5 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 思路:做细分，每个身体部位单独提取特征来作映射比例，这个比例代表了身体某部分在所有通道中的强弱，然后把比例成回对应的身体部分，再拼接起来
        # Input x = N C T V 32 64 64 25
        torso = x[:, :, :, [0, 1, 2, 3, 20]]   # torso = N C T V 32 64 64 5
        left_hand = x[:, :, :, [8, 9, 10, 11, 23, 24]]
        left_leg = x[:, :, :, [16, 17, 18, 19]]
        right_hand = x[:, :, :, [4, 5, 6, 7, 21, 22]]
        right_leg = x[:, :, :, [12, 13, 14, 15]]
        # body = {torso, left_hand, left_leg, right_hand, right_leg}
        avg_out_1 = self.fc_1(self.avg_pool(torso))  # N C T V 32 64 1 1
        max_out_1 = self.fc_1(self.max_pool(torso))
        avg_out_2 = self.fc_2(self.avg_pool(torso))  # N C T V 32 64 1 1
        max_out_2 = self.fc_2(self.max_pool(torso))
        avg_out_3 = self.fc_3(self.avg_pool(torso))  # N C T V 32 64 1 1
        max_out_3 = self.fc_3(self.max_pool(torso))
        avg_out_4 = self.fc_4(self.avg_pool(torso))  # N C T V 32 64 1 1
        max_out_4 = self.fc_4(self.max_pool(torso))
        avg_out_5 = self.fc_5(self.avg_pool(torso))  # N C T V 32 64 1 1
        max_out_5 = self.fc_5(self.max_pool(torso))

        out_part_1 = avg_out_1 + max_out_1
        out_part_1 = self.sigmoid(out_part_1)
        torso = torso * out_part_1

        out_part_2 = avg_out_2 + max_out_2
        out_part_2 = self.sigmoid(out_part_2)
        left_hand = left_hand * out_part_2

        out_part_3 = avg_out_3 + max_out_3
        out_part_3 = self.sigmoid(out_part_3)
        left_leg = left_leg * out_part_3

        out_part_4 = avg_out_4 + max_out_4
        out_part_4 = self.sigmoid(out_part_4)
        right_hand = right_hand * out_part_4

        out_part_5 = avg_out_5 + max_out_5
        out_part_5 = self.sigmoid(out_part_5)
        right_leg = right_leg * out_part_5

        out = torch.cat([torso, left_hand], dim=3)
        out = torch.cat([out, left_leg], dim=3)
        out = torch.cat([out, right_hand], dim=3)
        out = torch.cat([out, right_leg], dim=3)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # N C T V 32 1 64 25
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, block_size=41, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, aug=False):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)

        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, num_point, block_size, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, num_point, block_size, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, num_point, block_size, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, num_point, block_size, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, num_point, block_size, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, num_point, block_size, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, num_point, block_size, adaptive=adaptive, aug=True)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, num_point, block_size, stride=2, adaptive=adaptive, aug=True)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, num_point, block_size, adaptive=adaptive, aug=True)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, num_point, block_size, adaptive=adaptive, aug=True)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, keep_prob=0.9):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.dropT_skip(self.dropSke(x, keep_prob, self.A), keep_prob)
        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, 1.0)
        x = self.l8(x, 1.0)
        x = self.l9(x, 1.0)
        x = self.l10(x, 1.0)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
