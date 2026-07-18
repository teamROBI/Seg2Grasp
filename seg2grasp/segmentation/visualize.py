"""Lightweight mask/grasp visualization (replaces the legacy `visualize_ours`).

Pure OpenCV so it carries no detectron2 dependency.
"""
import cv2
import numpy as np

# A fixed, high-contrast color cycle (BGR) for instance masks.
_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
], dtype=np.uint8)[:, ::-1]  # RGB -> BGR


def overlay_masks(image, masks, bboxes=None, labels=None, alpha=0.25, outline=True):
    """Blend colored instance masks over ``image`` as light fill + crisp outlines.

    Args:
        image (np.ndarray): BGR image [H, W, 3].
        masks (np.ndarray | list): N boolean masks [H, W].
        bboxes (np.ndarray, optional): N boxes; drawn only if provided.
        labels (list[str], optional): per-mask text labels.
        alpha (float): fill strength (light by default so clutter stays readable).
        outline (bool): draw each instance's colored contour for separation.

    Returns:
        np.ndarray: a new BGR image with the overlay.
    """
    vis = image.copy()
    overlay = vis.copy()
    for i, mask in enumerate(masks):
        color = tuple(int(c) for c in _PALETTE[i % len(_PALETTE)])
        overlay[np.asarray(mask, dtype=bool)] = color
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    for i, mask in enumerate(masks):
        color = tuple(int(c) for c in _PALETTE[i % len(_PALETTE)])
        if outline:
            m = np.ascontiguousarray(mask, dtype=np.uint8)
            cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(vis, cnts, -1, color, 2)
        if bboxes is not None:
            x0, y0, x1, y1 = np.round(bboxes[i]).astype(int)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        if labels is not None and i < len(labels) and labels[i]:
            x0, y0 = (np.round(bboxes[i][:2]).astype(int) if bboxes is not None else (5, 15 + 18 * i))
            cv2.putText(vis, str(labels[i]), (int(x0), max(int(y0) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


def annotate_result(image, result, pc):
    """Standard Seg2Grasp overlay for a pipeline result.

    Colored mask outlines + light fill for every instance, a white outline and
    label on the chosen target, and the suction contact point (projected from the
    3-D suction point to the nearest pixel of ``pc``). Shared by the offline demo
    and the live loop so both look the same.
    """
    vis = overlay_masks(image, result.masks)                        # masks only, no boxes
    tmask = np.asarray(result.masks[result.target.index], np.uint8)
    cnts = cv2.findContours(tmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)             # white outline = target
    d = np.abs(pc - result.suction.point).sum(axis=2)
    v, u = np.unravel_index(np.argmin(d), d.shape)
    vis = draw_suction_point(vis, (u, v))
    label = result.label or "target"
    if result.label and result.confidence is not None:
        label = f"{result.label} ({result.confidence:.2f})"
    ys, xs = np.where(tmask)
    if len(xs):
        cv2.putText(vis, label, (max(int(xs.mean()) - 20, 3), max(int(ys.min()) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_suction_point(image, uv, color=(0, 255, 255), radius=6):
    """Draw the suction contact point at pixel ``uv=(u, v)``."""
    vis = image.copy()
    cv2.circle(vis, (int(uv[0]), int(uv[1])), radius, color, -1)
    cv2.circle(vis, (int(uv[0]), int(uv[1])), radius + 3, (0, 0, 0), 1)
    return vis
