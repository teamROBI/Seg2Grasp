#!/usr/bin/env bash
# Fetch the Seg2Grasp segmentation checkpoint from Hugging Face.
#
# Downloads model_final.pth (~2.5 GB) + config.yaml into
# data/checkpoints/segmentation/. The Qwen classifier weights download
# automatically from Hugging Face on first use (see scripts/serve_qwen.py --model).
#
#   bash scripts/download_weights.sh
#
# Source repo defaults to the one below; override with SEG2GRASP_WEIGHTS_HF:
#   SEG2GRASP_WEIGHTS_HF="<user>/<repo>" bash scripts/download_weights.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DST="$REPO/data/checkpoints/segmentation"
HF="${SEG2GRASP_WEIGHTS_HF:-jkim50104/Seg2Grasp}"
BASE="https://huggingface.co/${HF}/resolve/main"
mkdir -p "$DST"

echo ">>> Downloading segmentation checkpoint from https://huggingface.co/${HF} ..."
curl -L --fail "${BASE}/model_final.pth" -o "$DST/model_final.pth"
curl -L --fail "${BASE}/config.yaml"     -o "$DST/config.yaml"

if [[ -f "$DST/model_final.pth" && -f "$DST/config.yaml" ]]; then
    echo ">>> Done: $DST/{model_final.pth, config.yaml}"
else
    echo "!!! Download failed — check that ${HF} exists and its files are public."
    exit 1
fi
