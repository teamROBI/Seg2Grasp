"""Class-agnostic object segmentation (paper Section IV-A).

Wraps the Mask2Former (Swin-L) mask-proposal network trained on UOAIS-Sim. The
network is fed the RGB image and an inverted, normalized depth image; the model
runs its shared backbone on each and averages the outputs ("CFS" late fusion,
paper Eq. 1 / Fig. 3). Output is a set of class-agnostic instance masks.

The heavy model code (the ``mask2former`` package) is vendored under
``third_party/`` and detectron2 + the MSDeformAttn CUDA op are installed by
``scripts/setup_env.sh``. Inference builds the model directly and injects the
depth image into the input dict, so no detectron2 ``DefaultPredictor`` patch is
needed.

    seg = Segmenter()                          # uses paths.SEG_CONFIG / SEG_WEIGHTS
    masks, bboxes, vis = seg.segment(rgb, depth)
"""
import os
import sys

import cv2
import numpy as np

from .. import paths
from ..grasping.depth_utils import inpaint_depth, invert_depth
from .postprocess import postprocess_masks
from .visualize import overlay_masks

# Make the vendored `mask2former` package importable.
_THIRD_PARTY = os.path.join(paths.REPO_ROOT, "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)


def _boxes_from_masks(masks):
    """Derive [x0, y0, x1, y1] boxes from binary masks.

    Mask2Former is mask-based and its ``pred_boxes`` are not populated, so boxes
    are computed from each mask's pixel extent (as in the original pipeline).
    """
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if len(xs):
            boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return boxes


def _normalize_depth_for_seg(depth, min_val=300.0, max_val=1800.0):
    """Normalize raw metric depth (mm) to an 8-bit 3-channel image, robustly
    clipped to the 5-95 percentile band (matches the legacy inference)."""
    depth = depth.astype(np.float32)
    lo = max(np.percentile(depth, 5), min_val)
    hi = min(np.percentile(depth, 95), max_val)
    depth = np.clip(depth, lo, hi)
    depth = (depth - lo) / (hi - lo + 1e-8) * 255
    depth = np.expand_dims(depth, -1)
    return np.uint8(np.repeat(depth, 3, -1))


class Segmenter:
    """Mask2Former class-agnostic instance segmenter (RGB + inverted depth)."""

    def __init__(self, config_path=None, weights_path=None, score_threshold=0.5,
                 device=None):
        import torch
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        import detectron2.data.transforms as T
        from mask2former import add_maskformer2_config

        self._torch = torch
        config_path = config_path or paths.SEG_CONFIG
        weights_path = weights_path or paths.SEG_WEIGHTS
        self.score_threshold = score_threshold

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        if device is not None:
            cfg.MODEL.DEVICE = device
        cfg.freeze()
        self.cfg = cfg

        self.input_type = cfg.INPUT.INPUT_TYPE          # "rgb" | "rgbd" | "depth"
        self.depth_inverted = cfg.INPUT.DEPTH_INVERTED
        self.input_format = cfg.INPUT.FORMAT            # "RGB"
        self.device = cfg.MODEL.DEVICE

        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(weights_path)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

    # --- input prep ---
    def _prep_depth(self, depth):
        """Raw metric depth (mm) -> normalized, inpainted, inverted 3-channel image."""
        if depth.ndim == 2:
            depth_img = _normalize_depth_for_seg(depth)
        else:
            depth_img = depth  # already a normalized 3-channel depth image
        depth_img = inpaint_depth(depth_img)
        if self.depth_inverted:
            depth_img = invert_depth(depth_img)
        return depth_img

    def _to_tensor(self, image, flip_channels):
        """Apply the resize aug and convert an HWC uint8 image to a CHW float tensor."""
        if flip_channels:                    # BGR -> RGB when cfg.INPUT.FORMAT == "RGB"
            image = image[:, :, ::-1]
        image = self.aug.get_transform(image).apply_image(image)
        return self._torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)).astype("float32"))

    # --- inference ---
    def segment(self, rgb, depth=None, return_vis=True):
        """Segment a scene into class-agnostic instance masks.

        Args:
            rgb (np.ndarray): color image [H, W, 3] (BGR).
            depth (np.ndarray): raw metric depth [H, W] (mm) or a normalized
                3-channel depth image. Required for the rgbd/depth models.
            return_vis (bool): also return a mask-overlay visualization.

        Returns:
            (masks, bboxes, vis): masks [N, H, W] bool, bboxes [N, 4], and an
            overlay image (or None). Empty arrays if nothing is detected.
        """
        torch = self._torch
        H, W = rgb.shape[:2]
        flip = (self.input_format == "RGB")

        if self.input_type == "depth":
            image_t = self._to_tensor(self._prep_depth(depth), flip_channels=False)
            inputs = {"image": image_t, "height": H, "width": W}
        else:
            image_t = self._to_tensor(rgb, flip_channels=flip)
            inputs = {"image": image_t, "height": H, "width": W}
            if self.input_type == "rgbd":
                if depth is None:
                    raise ValueError("rgbd segmenter requires a depth image")
                inputs["depth_image"] = self._to_tensor(self._prep_depth(depth), flip_channels=False)

        with torch.no_grad():
            outputs = self.model([inputs])[0]
        instances = outputs["instances"].to("cpu")
        instances = instances[instances.scores > self.score_threshold]

        masks = instances.pred_masks.numpy()
        if len(masks) == 0:
            return np.empty((0, H, W), bool), np.empty((0, 4)), (rgb.copy() if return_vis else None)
        # Mask2Former is mask-based; derive boxes from the masks (pred_boxes are empty).
        bboxes = _boxes_from_masks(masks)

        masks, bboxes = postprocess_masks(masks, bboxes, W, H, rgb)
        vis = overlay_masks(rgb, masks, bboxes) if return_vis else None
        return masks.astype(bool), bboxes, vis
