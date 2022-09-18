import os
import numpy as np
import argparse
import torch

from models.network import get_network
from benchmark_aflw2000 import ana as ana_aflw2000, calc_nme as calc_nme_aflw2000
from benchmark_aflw import ana as ana_aflw, calc_nme as calc_nme_aflw
from utils.util import get_config
from utils.utils_inference import process_input


def cal_nme_with_68kp(net, face_detector, kpt_filter, root, flist):
    pts68_fit_all = []
    null_index = []
    kpt = np.zeros((2, 68))
    for i, img_file in enumerate(flist):
        img_path = os.path.join(root, img_file)
        vertices = process_input(img_path, net, face_detector)
        if vertices is None:
            pts68_fit_all.append(kpt)  # append the previous keypoint temporary work around
            null_index.append(i)
            continue

        kpt = vertices[kpt_filter, :]  # 68,3
        kpt = kpt.transpose()  # 3,68
        kpt = kpt[:2, :]  # 2,68

        pts68_fit_all.append(kpt)

    return pts68_fit_all, null_index


def main(args):
    cfg = get_config(args.configfile)
    local_rank = args.local_rank

    kpt_filter = np.loadtxt(cfg.filtered_kpt_500).astype(int)

    # ---- load detectors
    if cfg.is_dlib:
        import dlib
        detector_path = 'data/net-data/mmod_human_face_detector.dat'
        face_detector = dlib.cnn_face_detection_model_v1(
            detector_path)

    # load PRNet model
    net = get_network(cfg).to(local_rank)
    # print(net)
    net.load_state_dict(torch.load('checkpoints/net_39.pth'))
    net.eval()

    with torch.no_grad():
        # Calculate NME AFLW2000-3D
        root_aflw2000 = '/mnt/sata/code/myGit/3DFaceLight/prnet_tf/data/test.data/AFLW2000-3D_crop'
        img_list_aflw2000 = '/mnt/sata/code/myGit/3DFaceLight/prnet_tf/data/test.data/AFLW2000-3D_crop.list'
        with open(img_list_aflw2000) as f:
            flist_aflw2000 = [line.rstrip() for line in f]
        pts68_fit_all_aflw2000, null_index_aflw2000 = cal_nme_with_68kp(net, face_detector, kpt_filter,
                                                                        root_aflw2000, flist_aflw2000)
        ana_aflw2000(calc_nme_aflw2000(pts68_fit_all_aflw2000), null_index_aflw2000)

        # Calculate NME AFLW
        root_aflw = '/mnt/sata/code/myGit/3DFaceLight/prnet_tf/data/test.data/AFLW_GT_crop'
        img_list_aflw = '/mnt/sata/code/myGit/3DFaceLight/prnet_tf/data/test.data/AFLW_GT_crop.list'
        with open(img_list_aflw) as f:
            flist_aflw = [line.rstrip() for line in f]
        pts68_fit_all_aflw, null_index_aflw = cal_nme_with_68kp(net, face_detector, kpt_filter,
                                                                root_aflw, flist_aflw)
        ana_aflw(calc_nme_aflw(pts68_fit_all_aflw), null_index_aflw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--configfile', default='configs/config.py', help='path to the configfile')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    print(os.getcwd())
    args = parser.parse_args()
    main(args)
