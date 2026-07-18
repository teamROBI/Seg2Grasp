#!/usr/bin/env bash
# Set up the SEGMENTATION + GRASPING environment for Seg2Grasp, using uv.
#
# Runs the Mask2Former (Swin-L) mask-proposal network and the analytic suction
# planner. Python 3.10 / PyTorch 2.x (cu128) / detectron2 built from source —
# the same modern stack as the DA-Fusion reimplementation. The checkpoint was
# trained on torch 1.9, but its weights load fine on torch 2.x. It compiles two
# native extensions: detectron2's `_C` and the Mask2Former MSDeformAttn op.
#
# cu128 + arch 12.0 target NVIDIA Blackwell (sm_120) GPUs; adjust CUDA_ARCH /
# the torch index for other hardware. Requires `uv` and a matching CUDA toolkit
# (set CUDA_HOME) for the native compiles.
#
# Usage:  bash scripts/setup_seg_env.sh
# Result: a venv at venvs/seg. Activate with `source venvs/seg/bin/activate`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/venvs/seg"
CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-12.0}"   # Blackwell sm_120

echo ">>> Creating Python 3.10 venv at $VENV"
uv venv --python 3.10 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">>> Installing PyTorch (cu128)"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ">>> Installing build tools + Seg2Grasp runtime deps (no transformers here)"
uv pip install setuptools wheel ninja
uv pip install "numpy<2" "opencv-python>=4.9" "open3d>=0.18" "pillow>=10.0" \
    "scipy>=1.11" "scikit-image>=0.22" "matplotlib>=3.7" tqdm pyyaml imageio \
    timm shapely pycocotools "fvcore>=0.1.5" "iopath>=0.1.9" cython

echo ">>> Installing detectron2 from source (compiles _C for sm_${CUDA_ARCH})"
TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" FORCE_CUDA=1 \
    uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

echo ">>> Compiling MSDeformAttn CUDA op (needs CUDA_HOME set)"
( cd "$REPO/third_party/mask2former/modeling/pixel_decoder/ops" && \
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" FORCE_CUDA=1 python setup.py build install )

echo ">>> Installing seg2grasp (editable)"
uv pip install -e "$REPO" --no-deps

echo ">>> Done. Segmentation env ready at $VENV"
echo "    Note: the vendored legacy Mask2Former targets detectron2 0.6; if imports"
echo "    fail against detectron2 main, apply the small API fixes flagged at first run."
echo "    Next: place the checkpoint in data/checkpoints/segmentation/ (model_final.pth + config.yaml)"
