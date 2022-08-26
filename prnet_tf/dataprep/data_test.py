import glob
import os
import numpy as np
from modules.cv_plot import plot_kpt, plot_vertices
import cv2
from scipy.io import loadmat


def plot_GT_data():
    resolution_op = 256

    datapath = '/mnt/sata/data/300W_LP_UV/AFW'
    uv_kpt_ind_2d = np.loadtxt('../data/uv-data/uv_kpt_ind.txt').astype(np.int32)
    uv_kpt_ind = np.loadtxt('../data/save-img/blender/vertices_68.txt').astype(np.int32)
    face_ind = np.loadtxt('../data/uv-data/face_ind.txt').astype(
                np.int32)  # get valid vertices in the pos map
    face_ind_500 = np.loadtxt('../data/save-img/blender/vertices_500_sel_from_blender.txt').astype(
                np.int32)  # get valid vertices in the pos map
    files = glob.glob(os.path.join(datapath, '*.npy'))

    pos = np.load(files[0])
    img = cv2.imread(files[0].replace('.npy', '.jpg'))

    kpt = pos[uv_kpt_ind_2d[1, :], uv_kpt_ind_2d[0, :], :]
    all_vertices = np.reshape(pos, [resolution_op ** 2, -1])
    face_vertices = all_vertices[face_ind, :]
    face_vertices_500 = face_vertices[face_ind_500, :]

    result_list = [img,
                   plot_kpt(img, kpt),
                   plot_vertices(img, face_vertices),
                   plot_vertices(img, face_vertices_500)]

    cv2.imshow('Input', result_list[0])
    cv2.imshow('68 kpt', result_list[1])
    cv2.imshow('All kpt', result_list[2])
    cv2.imshow('500 kpt', result_list[3])
    cv2.moveWindow('Input', 0, 0)
    cv2.moveWindow('68 kpt', 200, 0)
    cv2.moveWindow('All kpt', 400, 0)
    cv2.moveWindow('500 kpt', 600, 0)
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit()


def testing():
    vertices = np.loadtxt('../data/save-img/blender/vertices_500_sel_from_blender.txt').astype(np.int32)
    print(len(vertices))
    vertices = np.unique(vertices)
    print(len(vertices))
    np.savetxt('../data/save-img/blender/vertices_365_iter_sel_from_blender.txt', vertices)


def main():
    plot_GT_data()
    # bfm = loadmat('/home/shubhajit/Downloads/BFM_info.mat')
    # bfm_uv = loadmat('/home/shubhajit/Downloads/BFM/BFM_UV.mat')
    # print('test')


if __name__=='__main__':
    main()