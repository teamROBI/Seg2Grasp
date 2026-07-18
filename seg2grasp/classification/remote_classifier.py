"""Client proxy for the Qwen classification microservice.

Lets the pipeline (running in the segmentation env) use the Qwen classifier that
lives in a separate env, over HTTP. Drop-in for :class:`QwenClassifier`: it
exposes the same ``classify(scene_bgr, crop_bgr) -> dict`` method, so the
pipeline doesn't know or care whether classification is in-process or remote.

    clf = RemoteQwenClassifier("http://127.0.0.1:8765")
    pipe = Seg2GraspPipeline(segmenter, clf)

Start the service first with ``scripts/serve_qwen.py`` in the Qwen env.
"""
import base64
import json
import urllib.request

import cv2


def _encode_png(bgr):
    if bgr is None:
        return None
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode() if ok else None


class RemoteQwenClassifier:
    def __init__(self, url="http://127.0.0.1:8765", timeout=120):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self):
        with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout) as r:
            return json.loads(r.read())

    def classify(self, scene_bgr, crop_bgr):
        payload = json.dumps({
            "scene": _encode_png(scene_bgr),
            "crop": _encode_png(crop_bgr),
        }).encode()
        req = urllib.request.Request(
            f"{self.url}/classify", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())
