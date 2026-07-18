#!/usr/bin/env python3
"""Offline Seg2Grasp demo on a saved RGB-D frame (e.g. the OCBD dataset).

Runs the full pipeline — segmentation -> target selection -> suction planning ->
classification — on one frame and writes an annotated visualization. No robot or
camera needed.

    # segmentation + grasp only (no classifier):
    python scripts/run_demo.py --scene Non-YCB/scene_hole_gray_bin --frame 0 --no-classify

    # with a running Qwen service (started in the Qwen env, see serve_qwen.py):
    python scripts/run_demo.py --scene MIXED --frame 3 --qwen-url http://127.0.0.1:8765

    # with an in-process classifier (only if seg + Qwen share one env):
    python scripts/run_demo.py --scene YCB/scene_large_yellow_bin --frame 2

Default dataset root: data/datasets/OCBD (override with --data-root).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seg2grasp import paths
from seg2grasp.pipeline import Seg2GraspPipeline
from seg2grasp.segmentation.segmenter import Segmenter
from seg2grasp.segmentation.visualize import overlay_masks, draw_suction_point


def load_frame(data_root, scene, frame):
    """Load an OCBD frame: RGB (BGR), metric depth (mm, from the point cloud),
    and the organized point cloud [H, W, 3] (mm)."""
    scene_dir = os.path.join(data_root, scene)
    rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"img_{frame}.png"))
    pc = np.load(os.path.join(scene_dir, "pcd", f"img_{frame}.npy")).astype(np.float32)
    if rgb is None or pc is None:
        raise FileNotFoundError(f"Could not load frame {frame} from {scene_dir}")
    depth = pc[:, :, 2]  # metric depth (mm) from the point cloud z-channel
    return rgb, depth, pc


def pc_point_to_pixel(pc, point):
    """Nearest pixel (u, v) in the organized cloud to a 3-D point (mm)."""
    d = np.abs(pc - point).sum(axis=2)
    v, u = np.unravel_index(np.argmin(d), d.shape)
    return int(u), int(v)


def build_classifier(args):
    if args.no_classify:
        return None
    if args.qwen_url:
        from seg2grasp.classification.remote_classifier import RemoteQwenClassifier
        return RemoteQwenClassifier(args.qwen_url)
    from seg2grasp.classification.qwen_classifier import QwenClassifier
    return QwenClassifier(open_vocab=args.open_vocab)


def main():
    ap = argparse.ArgumentParser(description="Offline Seg2Grasp demo on an RGB-D frame.")
    ap.add_argument("--data-root", default=os.path.join(paths.DATASET_ROOT, "OCBD"))
    ap.add_argument("--scene", default="Non-YCB/scene_hole_gray_bin", help="scene subdir under data-root")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--no-classify", action="store_true", help="skip classification (seg + grasp only)")
    ap.add_argument("--qwen-url", default=None, help="URL of a running Qwen service (RemoteQwenClassifier)")
    ap.add_argument("--open-vocab", action="store_true", help="free-form labels (in-process classifier)")
    ap.add_argument("--visualize", action="store_true", help="show the Open3D grasp view")
    ap.add_argument("--out", default=None, help="output image path (default: outputs/demo_<scene>_<frame>.png)")
    args = ap.parse_args()

    rgb, depth, pc = load_frame(args.data_root, args.scene, args.frame)
    print(f">>> Frame {args.scene}/img_{args.frame}: rgb {rgb.shape}, pc z {pc[..., 2].min():.0f}-{pc[..., 2].max():.0f} mm")

    segmenter = Segmenter()
    classifier = build_classifier(args)
    pipe = Seg2GraspPipeline(segmenter, classifier)

    result = pipe.run(rgb, depth, pc, classify=not args.no_classify, visualize=args.visualize,
                      rng=np.random.default_rng(0))
    if result is None:
        print(">>> No graspable object found.")
        return 1

    u, v = pc_point_to_pixel(pc, result.suction.point)
    label = f"{result.label} ({result.confidence:.2f})" if result.label and result.confidence \
        else (result.label or "")
    print(f">>> Target #{result.target.index}  suction=({result.suction.point.round(1)}) mm  "
          f"normal={result.suction.normal.round(3)}  label={label!r}")

    vis = overlay_masks(rgb, result.masks, result.bboxes)
    labels = [""] * len(result.masks)
    labels[result.target.index] = label or "target"
    vis = overlay_masks(vis, [result.masks[result.target.index]],
                        result.bboxes[result.target.index:result.target.index + 1], labels=[labels[result.target.index]])
    vis = draw_suction_point(vis, (u, v))

    out = args.out or os.path.join(paths.OUTPUT_ROOT, f"demo_{args.scene.replace('/', '_')}_{args.frame}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, vis)
    print(f">>> Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
