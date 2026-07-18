"""Target-object selection from a set of masks (paper Algorithm 1).

Among all segmented object masks, pick the target ``T`` to grasp: the most
elevated object — i.e. the one whose 3-D centroid is nearest to the (top-down)
camera (smallest depth ``z``). The most elevated object is the most accessible
for suction, and picking it top-down keeps the bin from being disturbed.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TargetObject:
    """The selected target: its index into the mask list, bounding box
    ``[x_min, y_min, x_max, y_max]``, 3-D centroid (mm), and the cropped
    point cloud / mask (and optionally RGB) of just that object's region."""
    index: int
    bbox: np.ndarray
    centroid: np.ndarray
    pc_crop: np.ndarray
    mask_crop: np.ndarray
    rgb_crop: np.ndarray = None


def _mask_centroid_3d(mask, pc_img):
    """3-D centroid (mm) of a mask, sampled at the mask's pixel centroid.
    Returns None if the sampled point has invalid (zero) depth."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    cy, cx = int(ys.mean()), int(xs.mean())
    if not mask[cy, cx]:
        # centroid pixel falls outside a non-convex mask — use the nearest mask pixel
        k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        cy, cx = ys[k], xs[k]
    xyz = pc_img[cy, cx].astype(np.float32)
    if xyz[2] <= 0:
        return None
    return xyz


def select_target(masks, bboxes, pc_img, rgb_img=None):
    """Select the most elevated (nearest-camera) object as the grasp target.

    Args:
        masks (np.ndarray | list): N boolean masks [H, W].
        bboxes (np.ndarray): N boxes [x_min, y_min, x_max, y_max].
        pc_img (np.ndarray): organized point cloud [H, W, 3] (mm).
        rgb_img (np.ndarray, optional): scene image [H, W, 3] to crop too.

    Returns:
        TargetObject, or None if no mask has a valid 3-D centroid.
    """
    if len(masks) == 0:
        return None

    centroids, valid = [], []
    for i, mask in enumerate(masks):
        c = _mask_centroid_3d(np.asarray(mask, dtype=bool), pc_img)
        if c is None or np.sum(bboxes[i]) == 0:
            centroids.append(np.array([0.0, 0.0, np.inf]))  # never selected
            valid.append(False)
        else:
            centroids.append(c)
            valid.append(True)

    if not any(valid):
        return None

    centroids = np.array(centroids)
    idx = int(np.argmin(centroids[:, 2]))         # smallest depth = most elevated

    x0, y0, x1, y1 = np.round(bboxes[idx]).astype(int)
    mask = np.asarray(masks[idx], dtype=bool)
    return TargetObject(
        index=idx,
        bbox=np.array([x0, y0, x1, y1]),
        centroid=centroids[idx],
        pc_crop=pc_img[y0:y1, x0:x1],
        mask_crop=mask[y0:y1, x0:x1],
        rgb_crop=(rgb_img[y0:y1, x0:x1] if rgb_img is not None else None),
    )
