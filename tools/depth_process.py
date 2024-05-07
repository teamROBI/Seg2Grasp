import cv2
import numpy as np
# from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
#                           ColorImage, DepthImage, RgbdImage)
import pyk4a
from pyk4a import Config, PyK4A
import multiprocessing as mp
import scipy.io as scio


def normalize_depth(depth, min_val=100.0, max_val=150.0):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    min_val = np.min(depth)
    max_val = np.max(depth)
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth


def normalize_depth_weight(depth, min_val=0, max_val=255):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 1
    return 1 - depth


def normalize_depth_gqcnn(depth, min_val=350.0, max_val=800.0):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 1.0
    # depth = np.expand_dims(depth, axis=2)
    return depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """ inpaint the input depth where the value is equal to zero

    Args:
        depth ([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        factor (int, optional): resize factor in depth inpainting. Defaults to 4.
        kernel_size (int, optional): kernel size in depth inpainting. Defaults to 5.

    Returns:
        [np.uint8]: inpainted depth array [H, W, 3] (0 ~ 255)
    """

    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W // factor, H // factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth

if __name__ == "__main__":
    meta = scio.loadmat("/home/tidy/PycharmProjects/icra2024_baseline/suctionnet_baseline/neural_network/data/scenes/scene_0100/kinect/meta/0001.mat")
    print(meta['intrinsic_matrix'])

    depth_im_filename = "../gqcnn/data/examples/clutter/phoxi/fcgqcnn/depth_0.npy"
    segmask_filename = "../gqcnn/data/examples/clutter/phoxi/fcgqcnn/segmask_0.png"
    camera_intr_filename = "../gqcnn/data/calib/phoxi/phoxi.intr"

    # Read images.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
    depth_data = np.load(depth_im_filename)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)


    # Azure Kinect
    mp.set_start_method("spawn", force=True)
    k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
                synchronized_images_only=True,
                camera_fps=pyk4a.FPS.FPS_5,
            ))
    k4a.start()

    capture = k4a.get_capture()
    rgb_img = capture.color[:, :, :3]
    # rgb_img = cv2.resize(rgb_img, (W, H))
    depth = capture.transformed_depth
    print(type(depth[600,600]),depth[600,600],depth.shape)
    depth_img = normalize_depth_gqcnn(depth)
    print(type(depth_img[600,600,0]),depth_img[600,600,0], depth_img.shape)
    # depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    # depth_img = inpaint_depth(depth_img)


    print(type(depth_data[0,0,0]),depth_data.shape)
    print(type(depth_im), depth_im.shape)

    print(depth_data[150][300])

    print(depth_data[0][0][0])
    for i in range(depth_data.shape[0]):
        for j in range(depth_data.shape[1]):
            try:
                if depth_data[i][j] - depth_im[i][j] != 0.0:
                    print(depth_data[i][j] - depth_im[i][j])
            except:
                print(i, j)


