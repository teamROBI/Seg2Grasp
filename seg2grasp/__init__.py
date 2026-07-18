"""Seg2Grasp: modular suction grasping for bin picking.

Three modules — segmentation (Mask2Former), grasping (analytic RANSAC suction),
classification (Qwen VLM) — orchestrated by :class:`seg2grasp.pipeline.Seg2GraspPipeline`.
"""
__version__ = "1.0.0"
