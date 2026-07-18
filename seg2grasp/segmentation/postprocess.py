"""Post-processing for raw Mask2Former instance masks.

Cleans the network's mask proposals before grasping: drops tiny masks, merges
heavily-overlapping duplicates, and refines each mask to its main connected
component (removing stray blobs). Ported from the DA-Fusion legacy
``eval/post_process.py`` (pure NumPy/OpenCV — no detectron2 dependency).
"""
import cv2
import numpy as np

MIN_MASK_PIXELS = 10          # drop masks smaller than this many pixels
OVERLAP_MERGE_RATIO = 0.7     # merge j into i if it overlaps >70% of the smaller mask
REFINE_MIN_AREA = 500         # refine_masks: drop main contour smaller than this
REFINE_MAX_AREA = 40000       # refine_masks: drop main contour larger than this
COMPONENT_MIN_AREA = 150      # keep sub-contours larger than this ...
COMPONENT_MAX_DIST = 100      # ... and within this distance of the main contour


def postprocess_masks(masks, bboxes, w, h, rgb_img=None):
    """Full clean-up chain: size prune -> merge overlaps -> refine.

    Args:
        masks (np.ndarray): [N, H, W] boolean/int masks.
        bboxes (np.ndarray): [N, 4] boxes.
        w, h (int): image width/height.
        rgb_img (np.ndarray, optional): unused except for legacy visualization.

    Returns:
        (masks, bboxes): the cleaned masks and their boxes.
    """
    mask_sizes = [int(np.count_nonzero(m)) for m in masks]
    keep = np.where(np.array(mask_sizes) > MIN_MASK_PIXELS)
    masks, bboxes = masks[keep], bboxes[keep]

    mask_sizes = [int(np.count_nonzero(m)) for m in masks]
    masks, bboxes = merge_overlapping_masks(masks, mask_sizes, bboxes)
    masks, bboxes, _ = refine_masks(masks, bboxes)
    return masks, bboxes


def prune_invalid_masks(masks, min_area=2000, max_area=50000):
    """Drop masks whose largest contour is outside the [min_area, max_area] band."""
    prune = []
    for idx in range(len(masks)):
        m = np.ascontiguousarray(masks[idx], dtype=np.uint8)
        contours = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        max_cnt = max(contours, key=cv2.contourArea, default=0)
        area = cv2.contourArea(max_cnt)
        if area > max_area or area < min_area:
            prune.append(idx)
    return np.delete(masks, prune, axis=0)


def merge_overlapping_masks(masks, mask_sizes, bboxes, ratio=OVERLAP_MERGE_RATIO):
    """Merge masks that overlap more than ``ratio`` of the smaller mask's area."""
    merged = []
    for i in range(len(masks)):
        if i in merged:
            continue
        for j in range(i + 1, len(masks)):
            if j in merged:
                continue
            overlap = np.count_nonzero(np.logical_and(masks[i] != 0, masks[j] != 0))
            small_overlap = overlap / max(min(mask_sizes[i], mask_sizes[j]), 1)
            if small_overlap > ratio:
                masks[i] += masks[j]
                mask_sizes[i] = int(np.count_nonzero(masks[i]))
                merged.append(j)
    masks = np.delete(masks, merged, axis=0)
    bboxes = np.delete(bboxes, merged, axis=0)
    masks[np.where(masks != 0)] = 1
    return masks, bboxes


def refine_masks(masks, bboxes):
    """Keep each mask's main blob plus nearby components; drop out-of-range masks.

    Returns (masks, bboxes, centers) where ``centers`` is the pixel centroid of
    each kept mask's largest contour.
    """
    keep, centers = [], []
    for idx in range(len(masks)):
        prune_mask = np.zeros(masks[idx].shape[:2], dtype=np.uint8)
        original = np.ascontiguousarray(masks[idx], dtype=np.uint8)

        contours = cv2.findContours(original, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        max_cnt = max(contours, key=cv2.contourArea, default=0)
        x, y, bw, bh = cv2.boundingRect(max_cnt)
        max_center = np.array([x + bw / 2, y + bh / 2])
        area = cv2.contourArea(max_cnt)
        if area > REFINE_MAX_AREA or area < REFINE_MIN_AREA:
            continue

        for cnt in contours:
            if cv2.contourArea(cnt) > COMPONENT_MIN_AREA:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                cnt_center = np.array([cx + cw / 2, cy + ch / 2])
                if np.linalg.norm(max_center - cnt_center) < COMPONENT_MAX_DIST:
                    cv2.drawContours(prune_mask, [cnt], 0, 1, -1)
        prune_mask = cv2.bitwise_and(original, original, mask=prune_mask)

        masks[idx] = prune_mask.astype(bool)
        keep.append(idx)
        centers.append(max_center)

    return masks[keep], bboxes[keep], centers
