import argparse
import os
import time
from torchsummary import summary
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.util import get_config
from models.network import get_network
from lr_scheduler import get_scheduler
from utils.utils_logging import AverageMeter, init_logging
from datasets.wlpuv_dataset import wlpuvDatasets
from losses.wing_loss import WingLoss_1


def main(config_file):
    cfg = get_config(config_file)  # parse_configuration(config_file)

    world_size = 1
    local_rank = args.local_rank
    torch.cuda.set_device(0)

    print('Initializing dataset...')
    train_set = wlpuvDatasets(cfg)
    cfg.num_images = len(train_set)
    cfg.world_size = world_size

    total_batch_size = cfg.batch_size * cfg.world_size
    epoch_steps = cfg.num_images // total_batch_size
    cfg.warmup_steps = epoch_steps * cfg.warmup_epochs
    if cfg.max_warmup_steps > 0:
        cfg.warmup_steps = min(cfg.max_warmup_steps, cfg.warmup_steps)
    cfg.total_steps = epoch_steps * cfg.num_epochs
    if cfg.lr_epochs is not None:
        cfg.lr_steps = [m * epoch_steps for m in cfg.lr_epochs]
    else:
        cfg.lr_steps = None
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_set, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=cfg.batch_size,  # sampler=train_sampler,
        num_workers=0, pin_memory=False, drop_last=True)

    print('The number of training samples = {0}'.format(cfg.num_images))

    # starting_epoch = cfg.load_checkpoint + 1
    # num_epochs = cfg.max_epochs

    net = get_network(cfg).to(local_rank)
    summary(net, (3, 256, 256))

    net.train()

    if cfg.opt == 'sgd':
        opt = torch.optim.SGD(
            params=[
                {"params": net.parameters()},
            ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.opt == 'adam':
        opt = torch.optim.Adam(
            params=[
                {"params": net.parameters()},
            ],
            lr=cfg.lr)
    elif cfg.opt == 'adamw':
        opt = torch.optim.AdamW(
            params=[
                {"params": net.parameters()},
            ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = get_scheduler(opt, cfg)

    start_epoch = 0
    total_step = cfg.total_steps

    loss = {
        'Loss': AverageMeter(),
    }

    # l1loss = nn.L1Loss()

    global_step = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        running_loss = 0.0
        for step, value in enumerate(train_loader):
            global_step += 1
            img = value['Image'].to(local_rank)
            dloss = {}
            assert cfg.task == 0
            label_verts = value['vertices_filtered'].to(local_rank)
            label_kpt = value['kpt'].to(local_rank)
            # label_points2d = value['points2d'].to(local_rank)

            # zero the parameter gradients
            opt.zero_grad()

            # -------- forward --------
            preds = net(img)
            # pred_verts, pred_points2d = preds.split([1220 * 3, 1220 * 2], dim=1)
            pred_verts = preds.view(cfg.batch_size, 500, 3)
            kpt_filer_index = torch.tensor(np.loadtxt(cfg.filtered_kpt_500).astype(int))
            pred_kpt = pred_verts[:, kpt_filer_index, :]
            # pred_points2d = pred_points2d.view(cfg.batch_size, 1220, 2)
            L3 = WingLoss_1()
            # loss1 = F.l1_loss(pred_verts, label_verts)
            loss2 = F.mse_loss(pred_kpt, label_kpt)
            loss3 = L3(pred_verts, label_verts)

            loss = 1.5 * loss3  # + 0.2*loss2

            # -------- backward + optimize --------
            loss.backward()
            opt.step()
            scheduler.step()

            running_loss += loss.item() * cfg.batch_size
            # print('batch loss :', loss.item())

        epoch_loss = running_loss / cfg.num_images
        print('epoch: {0} -> loss: {1} -> running loss: {2}'.format(epoch, loss.item(), epoch_loss))

        save_filename = 'net_%s.pth' % epoch
        save_path = os.path.join('checkpoints', save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        net.to(local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--configfile', default='configs/config.py', help='path to the configfile')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    print(os.getcwd())
    args = parser.parse_args()

    main(args.configfile)
