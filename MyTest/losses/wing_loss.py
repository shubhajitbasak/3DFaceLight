# -*- coding: utf-8 -*-
# @Time    : 2019/9/9
# @Author  : Elliott Zheng
# @Email   : admin@hypercube.top

import math
import torch
from torch import nn
import numpy as np


class WingLoss(nn.Module):
    # prediction and target both have the shape BS x N x DM = 32 x 68 x 2
    def __init__(self, width=5, curvature=0.5):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)

    def forward(self, prediction, target):
        diff = target - prediction
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger] = loss[idx_bigger] - self.C
        loss = loss.mean()
        print(loss)
        return loss


# torch.log  and math.log is e based
class WingLoss_1(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss_1, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


if __name__ == "__main__":
    loss_func = WingLoss()
    y = torch.ones(2, 68, 64, 64)
    y_hat = torch.zeros(2, 68, 64, 64)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)