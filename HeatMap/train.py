import argparse
import os
import torch
from torch.utils.data import DataLoader
import datetime
from tqdm import tqdm
from time import sleep
from HeatMap.conf import conf

from utils.loss import loss_heatmap, adaptive_wing_loss
from MyTest.lr_scheduler import get_scheduler
# from utils.utils_logging import AverageMeter
from MyTest.datasets.wlpuv_dataset import wlpuvDatasets
from utils.heatmap_3d import heatmap2sigmoidheatmap, cross_loss_entropy_heatmap,\
    heatmap2topkheatmap, heatmap2softmaxheatmap, lmks2heatmap3d
# from losses.wing_loss import WingLoss
from models.networks import Model3DHeatmap


def main(config_file):
    now = datetime.datetime.now()
    chkFolder = now.strftime("%b%d")

    cfg = conf.config

    world_size = 1
    local_rank = args.local_rank
    torch.cuda.set_device(0)

    if not os.path.exists(os.path.join('checkpoints/heatmap_3d', chkFolder)):
        os.makedirs(os.path.join('checkpoints/heatmap_3d', chkFolder), exist_ok=True)

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

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=cfg.batch_size,  # sampler=train_sampler,
        num_workers=0, pin_memory=False, drop_last=True)

    print('The number of training samples = {0}'.format(cfg.num_images))

    net = Model3DHeatmap(cfg).to(local_rank)

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
    global_step = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for step, value in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                global_step += 1
                # Image
                img = value['Image'].to(local_rank)

                # keypoints denormalized
                lmksGT = value['vertices_filtered'].to(local_rank)
                lmksGT = lmksGT * 255
                label_kpt = value['kpt'].to(local_rank)

                # Generate GT heatmap by randomized rounding
                heatGT = lmks2heatmap3d(lmksGT, 16, cfg.random_round, cfg.random_round_with_gaussian)

                # -------- forward --------
                heatPRED, kpPRED = net(img)

                if cfg.random_round_with_gaussian:
                    heatPRED = heatmap2sigmoidheatmap(heatPRED.to('cpu'))
                    loss = adaptive_wing_loss(heatPRED, heatGT)

                elif cfg.random_round:  # Using cross loss entropy
                    heatPRED = heatPRED.to('cpu')
                    loss = cross_loss_entropy_heatmap(heatPRED, heatGT, pos_weight=torch.Tensor([args.pos_weight]))
                else:
                    # MSE loss
                    if (cfg.get_topk_in_pred_heats_training):
                        heatPRED = heatmap2topkheatmap(heatPRED.to('cpu'))
                    else:
                        heatPRED = heatmap2softmaxheatmap(heatPRED.to('cpu'))

                    # Loss
                    loss = loss_heatmap(heatPRED, heatGT)



                # -------- backward + optimize --------
                # zero the parameter gradients
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()

                running_loss += loss.item() * cfg.batch_size
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)

        epoch_loss = running_loss / cfg.num_images
        print('epoch: {0} -> loss: {1} -> running loss: {2}'.format(epoch, loss.item(), epoch_loss))

        save_filename = 'net_%s.pth' % epoch
        save_path = os.path.join('checkpoints/heatmap_3d', chkFolder, save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        net.to(local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--configfile', default='configs/config.py', help='path to the configfile')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    print(os.getcwd())
    args = parser.parse_args()

    main(args.configfile)