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


import numpy as np

vertices_68 = np.loadtxt('../data/vertices_68.txt').astype(int)
vertices_500 = np.loadtxt('../data/vertices_500_sel_from_blender.txt').astype(int)
filter_68_500 = np.loadtxt('../data/vertices_68_fil_500.txt').astype(int)
vertices_500_68 = vertices_500[filter_68_500]
# np.savetxt('../data/vertices_68.txt', vertices_500_68)
vertices_500_68.sort()
vertices_68.sort()
print(vertices_500_68)
print(vertices_68)