#!/usr/bin/env python3
"""Local Qwen classification microservice.

Runs in the Qwen env (venvs/qwen) and loads the Qwen-VL model ONCE, then serves
classify requests over HTTP so the main pipeline (in the segmentation env) can
call it via RemoteQwenClassifier without sharing a Python environment.

    source venvs/qwen/bin/activate
    python scripts/serve_qwen.py --host 127.0.0.1 --port 8765

Protocol (JSON over HTTP):
    POST /classify  {"scene": <b64 png>, "crop": <b64 png>}
                 -> {"class_name": str|null, "class_idx": int,
                     "confidence": float|null, "raw": str}
    GET  /health -> {"status": "ok", "categories": [...]}
"""
import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, ".")
from seg2grasp.classification.qwen_classifier import QwenClassifier  # noqa: E402
from seg2grasp import paths  # noqa: E402


def _decode_png(b64):
    if not b64:
        return None
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_handler(classifier):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "categories": classifier.categories})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/classify":
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                scene = _decode_png(req.get("scene"))
                crop = _decode_png(req.get("crop"))
                result = classifier.classify(scene, crop)
                self._send(200, result)
            except Exception as e:  # keep the service alive on a bad request
                self._send(500, {"error": str(e)})

        def log_message(self, *a):  # quiet
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser(description="Qwen classification microservice.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=None,
                    help="model alias (%s), a full HF repo id, or a local path (default: %s)"
                         % (", ".join(paths.QWEN_MODELS), paths.QWEN_DEFAULT_MODEL))
    ap.add_argument("--dtype", default="auto", help='"auto" (FP8-safe) or a torch dtype e.g. bfloat16')
    ap.add_argument("--open-vocab", action="store_true", help="free-form labels (no fixed list)")
    args = ap.parse_args()

    print(f">>> Loading Qwen classifier: model={paths.resolve_qwen_model(args.model)} "
          f"dtype={args.dtype} open_vocab={args.open_vocab} ...", flush=True)
    clf = QwenClassifier(model_path=args.model, dtype=args.dtype, open_vocab=args.open_vocab)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(clf))
    print(f">>> Qwen service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
