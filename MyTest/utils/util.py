# import yaml
import importlib
import os
import os.path as osp
import numpy as np
import pickle
import torch
from torch.nn import functional as F


# def load_yaml(load_path):
#     """load yaml file"""
#     with open(load_path, 'r') as f:
#         loaded = yaml.load(f, Loader=yaml.Loader)
#
#     return loaded

def get_config(config_file):
    # assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    #print('A:', config_file, temp_config_name, temp_module_name)
    # config1 = importlib.import_module("configs.base")
    # importlib.reload(config1)
    # cfg = config1.config
    #print('B1:', cfg)
    # print(os.getcwd())
    # print("configs.%s"%temp_module_name)
    config2 = importlib.import_module("configs.%s"%temp_module_name)
    importlib.reload(config2)
    cfg = config2.config
    # #reload(config2)
    # job_cfg = config2.config
    # #print('B2:', job_cfg)
    # cfg.update(job_cfg)
    # cfg.job_name = temp_module_name
    # #print('B:', cfg)
    # if cfg.output is None:
    #     cfg.output = osp.join('work_dirs', temp_module_name)
    # #print('C:', cfg.output)
    # # cfg.flipindex = np.load(cfg.flipindex_file)
    return cfg


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))
    device = heatmap3d.device

    accu_x = accu_x * torch.arange(width).float().to(device)[None, None, :]
    accu_y = accu_y * torch.arange(height).float().to(device)[None, None, :]
    accu_z = accu_z * torch.arange(depth).float().to(device)[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


def loss_heatmap(gt, pre):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Fast_Human_Pose_Estimation_CVPR_2019_paper.pdf
    \gt BxCx64x64
    \pre BxCx64x64
    """
    # nn.MSELoss()
    B, C, H, W = gt.shape
    gt = gt.view(B, C, -1)
    pre = pre.view(B, C, -1)
    loss = torch.sum((pre - gt) * (pre - gt), axis=-1)  # Sum square error in each heatmap
    loss = torch.mean(loss, axis=-1)  # MSE in 1 sample / batch over all heatmaps
    loss = torch.mean(loss, axis=-1)  # Avarage MSE in 1 batch (.i.e many sample)
    return loss


def cross_loss_entropy_heatmap(p, g, pos_weight=torch.Tensor([1])):
    """\ Bx 106x 256x256
    """
    BinaryCrossEntropyLoss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    B, C, W, H = p.shape

    loss = BinaryCrossEntropyLoss(p, g)

    return loss


def adaptive_wing_loss(y_pred, y_true, w=14, epsilon=1.0, theta=0.5, alpha=2.1):
    """
    \ref https://arxiv.org/pdf/1904.07399.pdf
    """
    # Calculate A and C
    p1 = (1 / (1 + (theta / epsilon) ** (alpha - y_true)))
    p2 = (alpha - y_true) * ((theta / epsilon) ** (alpha - y_true - 1)) * (1 / epsilon)
    A = w * p1 * p2
    C = theta * A - w * torch.log(1 + (theta / epsilon) ** (alpha - y_true))

    # Asolute value
    absolute_x = torch.abs(y_true - y_pred)

    # Adaptive wingloss
    losses = torch.where(theta > absolute_x, w * torch.log(1.0 + (absolute_x / epsilon) ** (alpha - y_true)),
                         A * absolute_x - C)
    losses = torch.sum(losses, axis=[2, 3])
    losses = torch.mean(losses)

    return losses  # Mean wingloss for each sample in batch


def heatmap2topkheatmap(heatmap, topk=7):
    """
    \ Find topk value in each heatmap and calculate softmax for them.
    \ Another non topk points will be zero.
    \Based on that https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
    """
    N, C, H, W = heatmap.shape

    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N, C, 1, -1)

    score, index = heatmap.topk(topk, dim=-1)
    score = F.softmax(score, dim=-1)
    heatmap = F.softmax(heatmap, dim=-1)

    # Assign non-topk zero values
    # Assign topk with calculated softmax value
    res = torch.zeros(heatmap.shape)
    res = res.scatter(-1, index, score)

    # Reshape to the original size
    heatmap = res.view(N, C, H, W)
    # heatmap = heatmap.view(N, C, H, W)

    return heatmap


def mean_topk_activation(heatmap, topk=7):
    """
    \ Find topk value in each heatmap and calculate softmax for them.
    \ Another non topk points will be zero.
    \Based on that https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
    """
    N, C, H, W = heatmap.shape

    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N, C, 1, -1)

    score, index = heatmap.topk(topk, dim=-1)
    score = F.sigmoid(score)

    return score


def heatmap2softmaxheatmap(heatmap):
    N, C, H, W = heatmap.shape

    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N, C, 1, -1)
    heatmap = F.softmax(heatmap, dim=-1)

    # Reshape to the original size
    heatmap = heatmap.view(N, C, H, W)

    return heatmap


def heatmap2sigmoidheatmap(heatmap):
    heatmap = F.sigmoid(heatmap)

    return heatmap