import time
import cv2
import os
import glob
import argparse
import numpy as np
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt

import utils.util

from utils.utils_inference import plot_kpt, plot_vertices, plot_pose_box
from utils.utils_inference import estimate_pose, process_input, get_vertices, get_landmarks
from utils.util import get_config

from models.network import get_network



def main(config_file):
    cfg = get_config(config_file)
    local_rank = args.local_rank

    # ---- load detectors
    if cfg.is_dlib:
        import dlib
        detector_path = 'data/net-data/mmod_human_face_detector.dat'
        face_detector = dlib.cnn_face_detection_model_v1(
            detector_path)

    # load PRNet model
    net = get_network(cfg).to(local_rank)
    # print(net)
    net.load_state_dict(torch.load('checkpoints/Sep17/net_39.pth'))
    net.eval()

    # pts68_all_ori = _load('data/test-data/AFLW2000-3D.pts68.npy')

    # evaluation
    if not cfg.use_cam:  # on test-img
        print("[*] Processing on images in {}. Press 's' to save result.".format(cfg.eval_img_path))
        img_paths = glob.glob(os.path.join(cfg.eval_img_path, '*.jpg'))
        for img_path in img_paths:
            print(img_path)

            img = cv2.imread(img_path)
            vertices = process_input(img_path, net, face_detector)
            if vertices is None:
                continue

            kpt_filter = np.loadtxt(cfg.filtered_kpt_500).astype(int)

            kpt = vertices[kpt_filter, :]

            result_list = [img,
                           plot_vertices(img, vertices),
                           plot_kpt(img, kpt)]

            cv2.imshow('Input', result_list[0])
            cv2.imshow('Sparse alignment', result_list[1])
            cv2.imshow('Sparse alignment GT', result_list[2])
            cv2.moveWindow('Input', 0, 0)
            cv2.moveWindow('Sparse alignment', 500, 0)
            cv2.moveWindow('Sparse alignment GT', 1000, 0)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()

    else:  # webcam demo
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        count = 1
        # filtered_indexs = np.loadtxt('data/save-img/blender/vertices_500_sel_from_blender.txt').astype(int)
        while (True):
            _, image = cap.read()

            pos = process_input(image, net, face_detector)
            fps_str = 'FPS: %.2f' % (1 / (time.time() - start_time))
            start_time = time.time()
            cv2.putText(image, fps_str, (25, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow('Input', image)
            cv2.moveWindow('Input', 0, 0)

            key = cv2.waitKey(1)
            if pos is None:
                cv2.waitKey(1)
                cv2.destroyWindow('Sparse alignment')
                cv2.destroyWindow('Dense alignment')
                cv2.destroyWindow('Pose')
                if key & 0xFF == ord('q'):
                    break
                continue

            else:
                vertices = pos
                kpt_filter = np.loadtxt(cfg.filtered_kpt_500).astype(int)

                kpt = vertices[kpt_filter, :]
                # camera_matrix, _ = estimate_pose(vertices)
                # vertices_filtered = vertices[filtered_indexs]

                result_list = [plot_kpt(image, kpt),
                               plot_vertices(image, vertices) #,
                               # plot_pose_box(image, camera_matrix, kpt)
                               ]

                cv2.imshow('Sparse alignment', result_list[0])
                cv2.imshow('Dense alignment', result_list[1])
                # cv2.imshow('Pose', result_list[2])
                cv2.moveWindow('Sparse alignment', 500, 0)
                cv2.moveWindow('Dense alignment', 1000, 0)

                if key & 0xFF == ord('q'):
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--configfile', default='configs/config.py', help='path to the configfile')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    print(os.getcwd())
    args = parser.parse_args()

    main(args.configfile)
