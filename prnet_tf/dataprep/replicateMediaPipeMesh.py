import glob
import os
import cv2
import numpy as np
from prnet_tf.modules.cv_plot import plot_kpt, plot_vertices
import math
from typing import Tuple, Union
from scipy.spatial import cKDTree

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8)

mp_drawing = mp.solutions.drawing_utils


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):

    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def get_landmarks_mp(img, face_landmarks):
    image_rows, image_cols, _ = img.shape
    vert = []
    for landmark in face_landmarks.landmark:
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        vert.append(landmark_px)

    return np.asarray(vert)


def get_vertices(pos):
    resolution_op = 256
    face_ind = np.loadtxt('../data/uv-data/face_ind.txt').astype(
        np.int32)
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [resolution_op ** 2, -1])
    vertices = all_vertices[face_ind, :]

    return vertices


def get_index(vertices_all_2d, vertices_2d_new):
    ctree = cKDTree(vertices_all_2d)
    indexs = []
    for vert in vertices_2d_new:
        ds, inds = ctree.query(vert, 1)
        # print(ds, inds)
        indexs.append(inds)
    # ***********End Calculate Neighbour***********

    indexs = np.asarray(indexs)
    return indexs


img_path = '../data/save-img/blender/Iter2/images'
filter_520 = np.loadtxt('../data/save-img/blender/Iter2/'
                        'vertices_520_sel_from_blender.txt').astype(np.int32)
filter_68 = np.loadtxt('../data/save-img/blender/Iter2/'
                        'vertices_68.txt').astype(np.int32)
img_paths = glob.glob(os.path.join(img_path, '*.jpg'))
for img_path in img_paths:
    img = cv2.imread(img_path)
    pos = np.load(img_path.replace('.jpg', '.npy'))

    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face_landmarks = results.multi_face_landmarks[0]

    vertices_mp = get_landmarks_mp(img, face_landmarks)

    vertices_all = get_vertices(pos)
    vertices_filtered = vertices_all[filter_520]

    vertices_all_2d = vertices_all[:, :2]

    indices = get_index(vertices_all_2d, vertices_mp)

    indices_final = np.concatenate((indices, filter_68), axis=0)
    indices_final_unique = np.unique(indices_final, axis=0)

    vertices_filtered_mp = vertices_all[indices_final_unique]

    result_list = [img,
                   # plot_vertices(img, vertices),
                   plot_vertices(img, vertices_filtered_mp),
                   plot_vertices(img, vertices_filtered)]
    cv2.imshow('Input', result_list[0])
    cv2.imshow('Sparse key points MP', result_list[1])
    cv2.imshow('Sparse key points Blender', result_list[2])
    # cv2.imshow('mp result', result_list[3])
    cv2.moveWindow('Input', 0, 0)
    cv2.moveWindow('Sparse key points MP', 500, 0)
    cv2.moveWindow('Sparse key points Blender', 1000, 0)
    # cv2.moveWindow('mp result', 1500, 0)

    key = cv2.waitKey(0)
    if key == ord('q'):
        exit()
