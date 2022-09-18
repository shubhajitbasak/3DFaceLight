import scipy.io as sio
import cv2
from plot_kp import plot_kpt_2d

mat = sio.loadmat('/mnt/sata/data/300W_LP/landmarks/AFW/AFW_134212_1_0_pts.mat')
img = cv2.imread('/mnt/sata/data/300W_LP/AFW/AFW_134212_1_0.jpg')
kpt = mat['pts_2d']

cv2.imshow('Input', img)
cv2.imshow('Sparse alignment', plot_kpt_2d(img, kpt))
# cv2.imshow('Sparse alignment GT', cropped_img)
# cv2.imshow('Dense alignment', result_list[3])
cv2.moveWindow('Input', 0, 0)
cv2.moveWindow('Sparse alignment', 500, 0)
# cv2.moveWindow('Sparse alignment GT', 1000, 0)
# cv2.moveWindow('Dense alignment', 1500, 0)
key = cv2.waitKey(0)
if key == ord('q'):
    exit()