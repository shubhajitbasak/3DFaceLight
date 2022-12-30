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