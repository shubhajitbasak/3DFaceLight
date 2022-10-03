# import cv2
# import time
# cap = cv2.VideoCapture(0)
# start_time = time.time()
# count = 1
# # filtered_indexs = np.loadtxt('data/save-img/blender/vertices_500_sel_from_blender.txt').astype(int)
#
# _, image = cap.read()
#
# # pos = process_input(image, net, face_detector, cuda=isGPU)
# fps_str = 'FPS: %.2f' % (1 / (time.time() - start_time))
# start_time = time.time()
# cv2.putText(image, fps_str, (25, 25),
#             cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
# cv2.imshow('Input', image)
# cv2.moveWindow('Input', 0, 0)
#
# key = cv2.waitKey(1)
# if key & 0xFF == ord('q'):
#     exit()


# import numpy as np
#
# vertices_68 = np.loadtxt('../data/vertices_68.txt').astype(int)
# vertices_500 = np.loadtxt('../data/vertices_500_sel_from_blender.txt').astype(int)
# filter_68_500 = np.loadtxt('../data/vertices_68_fil_500.txt').astype(int)
# vertices_500_68 = vertices_500[filter_68_500]
# # np.savetxt('../data/vertices_68.txt', vertices_500_68)
# vertices_500_68.sort()
# vertices_68.sort()
# print(vertices_500_68)
# print(vertices_68)


## mediapipe testing
import numpy as np
import cv2
import mediapipe as mp
import math
from typing import Tuple, Union
from skimage.transform import estimate_transform, warp


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
        # to do: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return startX, startY, w, h


def crop_image(image, left, right, top, bottom, resolution_inp):
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
    size = int(old_size * 1.58)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    # image = image[:, :, ::-1] / 255.
    cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
    return cropped_image

resolution_inp = 256
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread('/mnt/sata/data/AFLW2000-3D/AFLW2000/image00013.jpg')
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


import dlib

detector_path_dlib = '../data/net-data/mmod_human_face_detector.dat'
face_detector_dlib = dlib.cnn_face_detection_model_v1(
    detector_path_dlib)
detected_faces_dlib = face_detector_dlib(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
d = detected_faces_dlib[
    0].rect

cropped_image = crop_image(image, d.left(), d.right(), d.top(), d.bottom(), resolution_inp)

cv2.imshow("Cropped_1", cropped_image)
cv2.moveWindow('Cropped_1', 0, 0)



# annotated_image1 = image.copy()
# x, y, w, h = convert_and_trim_bb(image, rect=d)
# cv2.rectangle(annotated_image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv2.imshow("Output_dlib", annotated_image1)
# cv2.moveWindow('Output_dlib', 500, 0)

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    if results.detections:
        for detection in results.detections:
            # mp_drawing.draw_detection(annotated_image, detection)

            image_rows, image_cols, _ = annotated_image.shape

            location = detection.location_data
            relative_bounding_box = location.relative_bounding_box

            rect_start_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                image_rows)
            rect_end_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                image_rows)

            # Flip the image horizontally for a selfie-view display.
            cv2.rectangle(annotated_image, rect_start_point, rect_end_point, (224, 224, 224), 2)
            cv2.imshow('Output_MP', annotated_image)
            cv2.moveWindow('Output_MP', 1000, 0)
cv2.waitKey(0)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_face_detection.FaceDetection(
#     model_selection=0, min_detection_confidence=0.5) as face_detection:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(image)
#
#     # Draw the face detection annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.detections:
#       for detection in results.detections:
#         mp_drawing.draw_detection(image, detection)
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()
