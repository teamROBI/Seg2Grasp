"""Depth-image utilities: normalize, invert, and inpaint.

Used to turn a raw metric depth map (mm) into the normalized/inverted 3-channel
image that the segmentation fusion input expects (paper Eq. 1), and to fill in
holes (zero-value pixels) left by the depth sensor.
"""
import cv2
import numpy as np


def normalize_depth(depth, min_val=None, max_val=None):
    """Normalize a metric depth map to an 8-bit 3-channel image (0-255).

    Args:
        depth (np.ndarray): depth array [H, W] in mm.
        min_val, max_val (float, optional): clip range; defaults to the
            per-image min/max.

    Returns:
        np.ndarray: normalized depth image [H, W, 3], uint8.
    """
    depth = depth.astype(np.float32)
    if min_val is None:
        min_val = np.min(depth)
    if max_val is None:
        max_val = np.max(depth)
    depth = np.clip(depth, min_val, max_val)
    depth = (depth - min_val) / (max_val - min_val + 1e-8) * 255
    depth = np.expand_dims(depth, -1)
    return np.uint8(np.repeat(depth, 3, -1))


def invert_depth(depth_uint8):
    """Invert a normalized 0-255 depth image so nearer objects appear brighter.

    This implements the depth inversion of paper Eq. 1 (S'_D = 1 - normalized S_D),
    which highlights closer objects and improves object/background contrast.
    """
    return 255 - depth_uint8


def normalize_depth_weight(depth, min_val=0, max_val=255):
    """Normalize depth to a 0-1 float weight map (1 = near, 0 = far)."""
    depth = depth.astype(np.float32)
    depth = np.clip(depth, min_val, max_val)
    depth = (depth - min_val) / (max_val - min_val + 1e-8)
    return 1 - depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """Inpaint (fill) zero-valued pixels of a normalized depth image.

    Args:
        depth (np.ndarray): normalized depth image [H, W, 3], uint8.
        factor (int): downscale factor applied before inpainting (speed).
        kernel_size (int): inpainting radius / dilation kernel size.
        dilate (bool): dilate the hole mask before inpainting.

    Returns:
        np.ndarray: inpainted depth image [H, W, 3].
    """
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W // factor, H // factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted = cv2.resize(inpainted, (W, H))
    return np.where(depth == 0, inpainted, depth)
