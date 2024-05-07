# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer, visualize_ours
from detectron2.utils.video_visualizer import VideoVisualizer
from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import torch
import pyk4a
from pyk4a import Config, PyK4A
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog

import imageio

# constants
WINDOW_NAME = "mask2former demo"

mp.set_start_method("spawn", force=True)
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
        synchronized_images_only=True,
        camera_fps=pyk4a.FPS.FPS_5,
    )
)

k4a.start()


def normalize_255(img, min_val=0, max_val=255*2):
    img = (img - min_val) / (max_val - min_val) * 255

    return img


def normalize_depth(depth, min_val=350.0, max_val=800.0):
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
    return depth

def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W // factor, H // factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth


def post_image_process(preds, bbox, w, h, rgb_img, visualize=True):
    # print("Original preds len:", len(preds))
    preds_len = [len(np.where(pred != 0)[0]) for pred in preds]
    preds = preds[np.where(np.array(preds_len) > 10)]
    # preds = prune_invalid(preds)
    # print("Delete nan preds len:", len(preds))

    preds_len = [len(np.where(pred != 0)[0]) for pred in preds]
    preds = merge_overlaps(preds, preds_len, w, h, visualize=False)
    # print("Overlap merged preds len:", len(preds))

    preds, bbox, max_center_list = refined_mask(preds, bbox, w, h, rgb_img, visualize=False)
    # print("Refined nan preds len:", len(preds))

    # preds, bbox = group_attention(preds, bbox, max_center_list)
    # print("Group attention preds len:", len(preds))

    return preds, bbox


def prune_invalid(preds):
    prune_keys = []
    for idx, pred in enumerate(preds):
        mask = np.ascontiguousarray(preds[idx], dtype=np.uint8)

        contours = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        max_cnt = max(contours, key=cv2.contourArea, default=0)  # max contour point
        if cv2.contourArea(max_cnt) > 50000 or cv2.contourArea(max_cnt) < 2000:
            prune_keys.append(idx)
            print("too large or small", cv2.contourArea(max_cnt))

    return np.delete(preds, prune_keys, axis=0)


def merge_overlaps(preds, preds_len, w, h, visualize=False):
    merged = []
    splited_pred = []
    for i in range(len(preds)):
        if i not in merged:
            for j in range(i + 1, len(preds)):
                if j not in merged:
                    overlapped = len(np.where(np.logical_and(preds[i] != 0, preds[j] != 0))[0])
                    small_overlap = overlapped / np.min([preds_len[i], preds_len[j]])
                    large_overlap = overlapped / np.max([preds_len[i], preds_len[j]])
                    if 0.5 < small_overlap:
                        # print(small_overlap, large_overlap, overlapped, preds_len[i], preds_len[j])
                        # preds[i] = np.mean(np.concatenate(([preds[i]], [preds[j]]), axis=0), axis=0)
                        preds[i] += preds[j]
                        preds_len[i] = len(np.where(preds[i] != 0)[0])
                        merged.append(j)

            # max_merged = np.max(preds[i][np.where(preds[i] != 0)])
            # print("max merged:", max_merged)
            if visualize:
                cv2.imshow("overlapped check", normalize_depth(preds[i], min_val=0, max_val=max_merged))
                key = cv2.waitKey(0)
                if key == 27:
                    break

            # if max_merged > 10:
            #     split1, split2 = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
            #     split1[np.where(preds[i] >= 10)] = 1
            #     split2[np.where(preds[i] < 10)] = 1
            #     splited_pred.append(split1)
            #     splited_pred.append(split2)

    preds = np.delete(preds, merged, axis=0)
    preds[np.where(preds != 0)] = 1
    # print(preds.shape, np.array([split1]).shape)
    # preds = np.concatenate((preds, [split1], [split2]), axis=0)

    return preds


def refined_mask(preds, bbox, w, h, rgb_img, visualize=False):
    add_keys = []
    max_center_list = []
    for idx, pred in enumerate(preds):
        prune_mask = np.zeros(pred.shape[:2], dtype=np.uint8)
        orignal_mask = np.ascontiguousarray(preds[idx], dtype=np.uint8)

        contours = cv2.findContours(orignal_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        max_cnt = max(contours, key=cv2.contourArea, default=0) # max contour point
        x, y, w, h = cv2.boundingRect(max_cnt)
        max_center = np.array([x + w / 2, y + h / 2])
        if cv2.contourArea(max_cnt) > 30000 or cv2.contourArea(max_cnt) < 500:
            print("too large or small", cv2.contourArea(max_cnt))
            continue
        else:
            cnt_dists = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 150:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cnt_center = np.array([x + w / 2, y + h / 2])
                    cnt_dists.append(np.linalg.norm(max_center - cnt_center))
                    if cnt_dists[-1] < 100:
                        cv2.drawContours(prune_mask, [cnt], 0, (1), -1)
                        # cv2.fillPoly(back, [cnt], (255, 255, 255))
            prune_mask = cv2.bitwise_and(orignal_mask, orignal_mask, mask=prune_mask) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            morphclose = cv2.morphologyEx(prune_mask, cv2.MORPH_CLOSE, kernel)

            preds[idx] = prune_mask.astype(bool)
            add_keys.append(idx)
            max_center_list.append(max_center)

            if visualize:
                print("contour size:", [cv2.contourArea(c) for c in contours], "contour dist from max:", cnt_dists)
                vis_img = np.hstack([orignal_mask*255, prune_mask, morphclose])
                cv2.imshow('Mask post processed', vis_img)

                back_rgb = cv2.cvtColor(prune_mask, cv2.COLOR_GRAY2RGB)
                back_rgb[np.where(prune_mask == 255)] = [0, 0, 255]
                rgb_img = cv2.addWeighted(rgb_img, 1, back_rgb, 0.8, 0.5)
                cv2.imshow('RGB Image', rgb_img)
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

    return preds[add_keys], bbox[add_keys], max_center_list


def group_attention(preds,bbox, max_center_list):
    prune_keys = []
    for i in range(len(max_center_list)):
        dist = []
        for j in range(len(max_center_list)):
            if i != j:
                dist.append(np.linalg.norm(max_center_list[i] - max_center_list[j]))
        if np.min(dist) > 150:
            prune_keys.append(i)
    #     print(np.min(dist))
    # print(prune_keys)
    return np.delete(preds, prune_keys, axis=0), np.delete(bbox, prune_keys, axis=0)


def average_filter(rgb_img, depth_img, kernel_size):
    # kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    # Pad the image to handle border pixels
    padding = kernel_size // 2
    padded_image = np.pad(rgb_img, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    padded_kernel = np.pad(depth_img, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    filtered_image = np.zeros_like(rgb_img, dtype=np.uint8)

    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            for c in range(rgb_img.shape[2]):
                # Apply the kernel to each channel of the image
                neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                kernel = padded_kernel[i:i + kernel_size, j:j + kernel_size, c]
                # kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
                kernel = np.where((kernel[1, 1] - 5 < kernel) & (kernel < kernel[1, 1] + 5), kernel, 0)
                kernel = kernel / np.sum(kernel)
                # print(kernel)
                filtered_image[i, j, c] = np.sum(neighborhood * kernel)

    return filtered_image


def set_cfg(config_path, weight_path):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_deeplab_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path  # 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_100ep/model_final_f07440.pkl'
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True

    return cfg

if __name__ == "__main__":
    # v = DemoDepth()
    config_path = "/home/tidy/PycharmProjects/icra2024_baseline/Mask2Former/weights/config.yaml"
    weight_path = "/home/tidy/PycharmProjects/icra2024_baseline/Mask2Former/weights/model_final.pth" # Model Change Needed
    crop_size = [160, 340, 400, 600]
    y_start, x_start, height, width = crop_size

    cfg = set_cfg(config_path, weight_path)
    predictor = DefaultPredictor(cfg)
    cpu_device = torch.device("cpu")
    rgbd_fusion = cfg.INPUT.FUSION_TYPE

    while True:
        # vis = v.video(cam)
        cam = k4a.get_capture()
        rgb_img = cam.color[:, :, :3][y_start:y_start + height, x_start:x_start + width]
        depth = cam.transformed_depth[y_start+3:y_start+3 + height, x_start+9:x_start+9 + width]
        depth_img = normalize_depth(depth)
        depth_img = inpaint_depth(depth_img)
        depth_img = 255 - depth_img

        if cfg.INPUT.CONTRAST:
            alpha = 0.5
            depth_img = np.clip((1 + alpha) * depth_img - 128 * alpha, 0, 255).astype(np.uint8)

        if rgbd_fusion == 'depth':
            our_input = depth_img
        elif rgbd_fusion == 'early':
            image = rgb_img.astype(np.int32) + depth_img.astype(np.int32)
            image_max, image_min = np.max(image), np.min(image)
            image = (image - image_min) / (image_max - image_min) * 255
            our_input = np.uint8(image)

        elif rgbd_fusion == 'add_weight':
            our_input = cv2.addWeighted(rgb_img, 0.5, depth_img, 0.5, 0)

        elif rgbd_fusion == 'abs_diff':
            our_input = cv2.absdiff(rgb_img, depth_img)

        elif rgbd_fusion == 'depth_weighted':
            weight_strength = cfg.INPUT.WEIGHT_STRENGTH
            depth_weight = depth_img.astype(np.float32) / 255
            img = rgb_img.astype(np.float32) * (depth_weight ** weight_strength)
            img_max, img_min = np.max(img), np.min(img)
            img = (img - img_min) / (img_max - img_min) * 255
            our_input = np.uint8(img)

        elif rgbd_fusion == 'late':
            our_input = np.concatenate([rgb_img, depth_img], -1)
        else:
            our_input = rgb_img


        predictions = predictor(our_input)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = rgb_img[:, :, ::-1]
        instances = predictions["instances"].to(cpu_device)
        preds = instances.pred_masks.detach().cpu().numpy()
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy()

        preds, bboxes = post_image_process(preds, bboxes, width, height, rgb_img)
        # break

        vis_img = visualize_ours(image, preds, bboxes)
        cv2.imshow('pred', vis_img[:, :, ::-1])

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
