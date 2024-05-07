import pyk4a
import multiprocessing as mp
import cv2
import numpy as np
import os
import seg2grasp
from tools.depth_process import *

server = True

if server:
    cap = cv2.VideoCapture(0)
else:
    mp.set_start_method("spawn", force=True)
    k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
                synchronized_images_only=True,
                camera_fps=pyk4a.FPS.FPS_5,
            ))

start_img = cv2.imread("/home/tidy/PycharmProjects/icra2024_baseline/start.jpg")
episode_idx = int(input("episode idx num: "))
crop_size = [230, 480, 370, 265]
y_start, x_start, height, width = crop_size

idx = 0
while True:
    if server:
        ret, img = cap.read()
    else:
        pass

    cv2.imshow('start venv3.8', start_img)
    key = cv2.waitKey(1)

    if key == ord('s'):
        k4a.start()

        capture = k4a.get_capture()
        rgb_img = capture.color[:, :, :3]
        depth = capture.transformed_depth

        depth = depth[:, :, 0]

        rgb_img = rgb_img[y_start:y_start + height, x_start:x_start + width]
        depth = depth[y_start:y_start + height, x_start:x_start + width]
        depth = depth[:, :, 0]

        print(depth.shape, np.min(depth), np.max(depth))

        pc_img = capture.transformed_depth_point_cloud[y_start:y_start + height, x_start:x_start + width]

        rgb_img = rgb_img.copy()
        w, h = rgb_img.shape[1], rgb_img.shape[0]

        ours_img = seg2grasp.detect(rgb_img.copy(), depth.copy(), w, h, pc_img.copy())

        cv2.imshow('Output', ours_img)
        key = cv2.waitKey(1)


        print(f'{idx} saved')
        idx += 1

        k4a.close()

    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()