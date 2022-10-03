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
import cv2
import mediapipe as mp
import math
from typing import Tuple, Union


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


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread('/mnt/sata/data/AFLW2000-3D/AFLW2000/image00013.jpg')
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
            cv2.imshow('detection', annotated_image)
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
