#!/usr/bin/env python3
"""Live Seg2Grasp loop: Azure Kinect -> pipeline -> (optional) robot grasp.

Captures frames from an Azure Kinect, runs the full pipeline, and prints/plots
the chosen suction pose and label each iteration. Robot actuation is OPTIONAL and
guarded — without a configured UR5e (seg2grasp/robot/, see its README) the loop
still runs as a perception demo (planning + visualization only).

    source venvs/seg/bin/activate
    # start the Qwen service in the other env first (optional):
    #   source venvs/qwen/bin/activate && python scripts/serve_qwen.py
    python scripts/run_live.py --qwen-url http://127.0.0.1:8765

Press 'q' in the view window to quit, any other key to grab the next frame.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seg2grasp.pipeline import Seg2GraspPipeline
from seg2grasp.segmentation.segmenter import Segmenter
from seg2grasp.segmentation.visualize import annotate_result


def main():
    ap = argparse.ArgumentParser(description="Live Seg2Grasp loop (Kinect + pipeline).")
    ap.add_argument("--qwen-url", default=None, help="URL of a running Qwen service")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--roi", default=None, help="bin crop 'y,x,h,w' (default: full frame)")
    ap.add_argument("--execute", action="store_true", help="actually command the robot (needs seg2grasp/robot)")
    args = ap.parse_args()

    from seg2grasp.camera.kinect import KinectCamera
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    camera = KinectCamera(roi=roi)

    segmenter = Segmenter()
    classifier = None
    if not args.no_classify and args.qwen_url:
        from seg2grasp.classification.remote_classifier import RemoteQwenClassifier
        classifier = RemoteQwenClassifier(args.qwen_url)
    pipe = Seg2GraspPipeline(segmenter, classifier)

    agent = None
    if args.execute:
        agent = _init_robot()  # optional; see seg2grasp/robot/README.md

    print(">>> Live loop. 'q' to quit.")
    try:
        while True:
            rgb, depth, pc = camera.capture()
            result = pipe.run(rgb, depth, pc, classify=not args.no_classify)
            if result is None:
                cv2.imshow("Seg2Grasp (q=quit)", rgb)
                if cv2.waitKey(0) in (ord("q"), 27):
                    break
                continue

            label = f"{result.label} ({result.confidence:.2f})" if result.label and result.confidence else (result.label or "")
            print(f"target #{result.target.index}  point={result.suction.point.round(1)} mm  "
                  f"normal={result.suction.normal.round(3)}  label={label!r}")

            vis = annotate_result(rgb, result, pc)
            cv2.imshow("Seg2Grasp (q=quit)", vis)
            if cv2.waitKey(0) in (ord("q"), 27):
                break

            if agent is not None:
                _execute_grasp(agent, result)
    finally:
        camera.close()
        cv2.destroyAllWindows()


def _init_robot():
    """Initialize the UR5e agent. Hardware-specific — see seg2grasp/robot/README.md."""
    raise NotImplementedError(
        "Robot execution requires the UR5e stack (seg2grasp/robot/, needs the `urx` package). "
        "Wire up Agent here for your cell, then remove this guard.")


def _execute_grasp(agent, result):
    """Convert the suction pose to a robot motion and pick. Fill in per your cell."""
    raise NotImplementedError("Implement camera->robot transform + move/grip using seg2grasp/robot/core/agent.py")


if __name__ == "__main__":
    main()
