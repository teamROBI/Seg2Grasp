#!/usr/bin/env python3
"""Seg2Grasp demo gallery — run the full pipeline over the bundled sample frames.

For every frame in ``demo/samples/`` this runs segmentation -> target selection ->
suction planning (-> classification when a Qwen service URL is given), writes an
annotated image per sample, and stitches them into a single ``gallery.png``.
Optionally emits the point-cloud "how the suction point is chosen" figure per
sample (``--grasp-steps``).

    source venvs/seg/bin/activate

    # segmentation + grasp (default):
    python demo/run.py

    # add the grasp-selection point-cloud figures:
    python demo/run.py --grasp-steps

    # full three-module run (start the Qwen service in the qwen env first):
    #   source venvs/qwen/bin/activate && python scripts/serve_qwen.py
    python demo/run.py --qwen-url http://127.0.0.1:8765

Outputs go to ``demo/outputs/``.
"""
import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from seg2grasp.pipeline import Seg2GraspPipeline                       # noqa: E402
from seg2grasp.segmentation.segmenter import Segmenter                 # noqa: E402
from seg2grasp.segmentation.visualize import annotate_result          # noqa: E402
from seg2grasp.grasping.suction_planner import estimate_suction_point, visualize_suction_steps  # noqa: E402

SAMPLES = os.path.join(REPO, "demo", "samples")
OUTDIR = os.path.join(REPO, "demo", "outputs")




def make_gallery(images, cols=6, pad=6, bg=32):
    """Stitch equal-size annotated images into a grid montage."""
    if not images:
        return None
    h = min(im.shape[0] for im in images)
    tiles = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images]
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(bg, bg, bg)) for t in tiles]
    rows = math.ceil(len(tiles) / cols)
    grid = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        grid[y:y + h, x:x + w] = t
    return grid


def main():
    ap = argparse.ArgumentParser(description="Seg2Grasp demo gallery over demo/samples/.")
    ap.add_argument("--qwen-url", default=None, help="URL of a running Qwen service (enables classification)")
    ap.add_argument("--grasp-steps", action="store_true", help="also save the suction-selection point-cloud figure per sample")
    ap.add_argument("--cols", type=int, default=6, help="columns in the gallery montage")
    args = ap.parse_args()

    samples = sorted(glob.glob(os.path.join(SAMPLES, "*")))
    if not samples:
        print(f"No samples in {SAMPLES}")
        return 1
    os.makedirs(OUTDIR, exist_ok=True)

    classifier = None
    if args.qwen_url:
        from seg2grasp.classification.remote_classifier import RemoteQwenClassifier
        classifier = RemoteQwenClassifier(args.qwen_url)

    segmenter = Segmenter()
    pipe = Seg2GraspPipeline(segmenter, classifier)

    gallery_imgs = []
    for d in samples:
        name = os.path.basename(d)
        rgb = cv2.imread(os.path.join(d, "rgb.jpg"))
        pc = np.load(os.path.join(d, "pcd.npz"))["pc"].astype(np.float32)
        result = pipe.run(rgb, pc[:, :, 2], pc, classify=bool(classifier),
                          rng=np.random.default_rng(0))
        if result is None:
            print(f"{name}: no graspable object")
            continue

        vis = annotate_result(rgb, result, pc)
        cv2.imwrite(os.path.join(OUTDIR, f"{name}.png"), vis)
        gallery_imgs.append(vis)
        lab = f" label={result.label}" if result.label else ""
        print(f"{name}: masks={len(result.masks)} target#{result.target.index} "
              f"suction={result.suction.point.round(0)}{lab}")

        if args.grasp_steps:
            dbg = {}
            t = result.target
            estimate_suction_point(t.pc_crop, mask=t.mask_crop, rng=np.random.default_rng(0), debug=dbg)
            visualize_suction_steps(dbg, save_path=os.path.join(OUTDIR, f"{name}_grasp_steps.png"),
                                    title=f"Suction selection — {name}")

    grid = make_gallery(gallery_imgs, cols=args.cols)
    if grid is not None:
        path = os.path.join(OUTDIR, "gallery.png")
        cv2.imwrite(path, grid)
        print(f"\nWrote {len(gallery_imgs)} results + gallery -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
