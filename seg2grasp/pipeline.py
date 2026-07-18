"""End-to-end Seg2Grasp pipeline: segment -> select target -> grasp -> classify.

This is the single entry point that ties the three modules together (the paper's
modular design). Modules are injected so each can be used, swapped, or tested
independently:

    seg = Segmenter()                 # segmentation/segmenter.py
    clf = QwenClassifier()            # classification/qwen_classifier.py
    pipe = Seg2GraspPipeline(seg, clf)
    result = pipe.run(rgb, depth, pc)

`run` returns a :class:`Seg2GraspResult` with the chosen target's suction pose
(3-D point + normal), its category label, and everything needed to visualize or
command the robot. Segmentation and classification are heavy (GPU); the grasping
stage is analytic. Classification can be skipped for a grasp-only run.
"""
from dataclasses import dataclass

import numpy as np

from .grasping.suction_planner import estimate_suction_point, SuctionPose
from .grasping.target_selection import select_target, TargetObject


@dataclass
class Seg2GraspResult:
    """Full pipeline output for one RGB-D frame."""
    target: TargetObject            # selected object (index, bbox, crops, centroid)
    suction: SuctionPose            # grasp point (mm) + camera-facing normal
    label: str = None               # predicted category (None if not classified)
    confidence: float = None        # classifier confidence
    masks: np.ndarray = None        # all segmentation masks [N, H, W]
    bboxes: np.ndarray = None       # all boxes [N, 4]
    seg_vis: np.ndarray = None      # segmentation visualization image (optional)


class Seg2GraspPipeline:
    """Orchestrates the three Seg2Grasp modules over a single RGB-D frame.

    Args:
        segmenter: object with ``segment(rgb, depth) -> (masks, bboxes, vis)``.
        classifier: object with ``classify(scene_bgr, crop_bgr) -> dict`` (may be
            None to skip classification).
        vacuum_radius (float): suction-cup radius (mm) passed to the grasp planner.
    """

    def __init__(self, segmenter, classifier=None, vacuum_radius=30.0):
        self.segmenter = segmenter
        self.classifier = classifier
        self.vacuum_radius = vacuum_radius

    def run(self, rgb, depth, pc, classify=True, visualize=False, rng=None):
        """Run the full pipeline on one frame.

        Args:
            rgb (np.ndarray): scene color image [H, W, 3] (BGR).
            depth (np.ndarray): raw metric depth [H, W] (mm) or normalized depth
                image, as expected by the segmenter.
            pc (np.ndarray): organized point cloud [H, W, 3] (mm).
            classify (bool): run the classifier on the chosen target.
            visualize (bool): show the Open3D grasp visualization.

        Returns:
            Seg2GraspResult, or None if no object could be segmented/selected.
        """
        masks, bboxes, seg_vis = self.segmenter.segment(rgb, depth)
        if masks is None or len(masks) == 0:
            return None

        target = select_target(masks, bboxes, pc, rgb_img=rgb)
        if target is None:
            return None

        suction = estimate_suction_point(
            target.pc_crop, mask=target.mask_crop, vacuum_radius=self.vacuum_radius,
            rng=rng, visualize=visualize, rgb_img=target.rgb_crop)
        if suction is None:
            return None

        label, confidence = None, None
        if classify and self.classifier is not None and target.rgb_crop is not None:
            pred = self.classifier.classify(rgb, target.rgb_crop)
            label, confidence = pred.get("class_name"), pred.get("confidence")

        return Seg2GraspResult(
            target=target, suction=suction, label=label, confidence=confidence,
            masks=masks, bboxes=bboxes, seg_vis=seg_vis)
