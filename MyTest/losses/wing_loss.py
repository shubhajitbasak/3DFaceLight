# -*- coding: utf-8 -*-
# @Time    : 2019/9/9
# @Author  : Elliott Zheng
# @Email   : admin@hypercube.top

import math
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import numpy as np


# class WingLoss(nn.Module):
#     # prediction and target both have the shape BS x N x DM = 32 x 68 x 2
#     def __init__(self, width=5, curvature=0.5):
#         super(WingLoss, self).__init__()
#         self.width = width
#         self.curvature = curvature
#         self.C = self.width - self.width * np.log(1 + self.width / self.curvature)
#
#     def forward(self, prediction, target):
#         diff = target - prediction
#         diff_abs = diff.abs()
#         loss = diff_abs.clone()
#
#         idx_smaller = diff_abs < self.width
#         idx_bigger = diff_abs >= self.width
#
#         loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
#         loss[idx_bigger] = loss[idx_bigger] - self.C
#         loss = loss.mean()
#         print(loss)
#         return loss


# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
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


class WingLoss_1(_Loss):
    # https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch/blob/master/utils/wing_loss.py
    def __init__(self, width=10, curvature=2.0, reduction="mean"):
        super(WingLoss_1, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return self.wing_loss(prediction, target, self.width, self.curvature, self.reduction)

    def wing_loss(self, prediction, target, width=10, curvature=2.0, reduction="mean"):
        diff_abs = (target - prediction).abs()
        loss = diff_abs.clone()
        idx_smaller = diff_abs < width
        idx_bigger = diff_abs >= width
        # loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)
        loss_smaller = width * torch.log(1 + diff_abs[idx_smaller] / curvature)
        C = width - width * math.log(1 + width / curvature)
        # loss[idx_bigger] = loss[idx_bigger] - C
        loss_biger = loss[idx_bigger] - C
        loss = torch.cat((loss_smaller, loss_biger), 0)
        if reduction == "sum":
            loss = loss.sum()
        if reduction == "mean":
            loss = loss.mean()
        return loss


'''
Adaptive Wing Loss from 
Wang X, Bo L, Fuxin L. Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression. ICCV2019.
The following module is based on https://github.com/protossw512/AdaptiveWingLoss
'''


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


if __name__ == "__main__":
    loss_func = AdaptiveWingLoss()
    y = torch.ones(2, 68, 64, 64)
    y_hat = torch.zeros(2, 68, 64, 64)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)
