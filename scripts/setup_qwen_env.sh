#!/usr/bin/env bash
# Set up the CLASSIFICATION environment for Seg2Grasp (Qwen VLM), using uv.
#
# Kept SEPARATE from the segmentation env: both are py3.10 / torch 2.x, but Qwen
# needs transformers>=5 while segmentation needs detectron2 + the MSDeformAttn op,
# and those heavy stacks are cleaner to isolate. This env runs the Qwen classifier
# — either in-process (standalone) or as a local microservice (scripts/serve_qwen.py)
# that the main pipeline calls via RemoteQwenClassifier.
#
# Requirements: `uv` (https://docs.astral.sh/uv/).
#
# Usage:  bash scripts/setup_qwen_env.sh
# Result: a venv at venvs/qwen. Activate with `source venvs/qwen/bin/activate`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/venvs/qwen"

echo ">>> Creating Python 3.10 venv at $VENV (uv fetches CPython 3.10 if needed)"
uv venv --python 3.10 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">>> Installing PyTorch (cu128; Blackwell/sm_120)"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ">>> Installing transformers 5 + deps for Qwen-VL"
# compressed-tensors + kernels enable the FP8 checkpoints (finegrained-fp8 kernel).
uv pip install setuptools wheel packaging ninja
uv pip install "transformers>=5" "accelerate>=1.0" "compressed-tensors" \
    "kernels>=0.15.2,<0.16.0" "pillow>=10.0" "opencv-python>=4.9" "numpy<2" pyyaml

# Fast path for Qwen's linear-attention (Gated DeltaNet) layers — WITHOUT these the
# model falls back to a slow torch implementation (~20x slower). --no-build-isolation
# so causal-conv1d compiles against the env's torch (needs CUDA_HOME set).
echo ">>> Installing fast kernels: flash-linear-attention + causal-conv1d"
uv pip install "flash-linear-attention==0.5.1"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
    uv pip install --no-build-isolation "causal-conv1d==1.6.2.post1"

echo ">>> Installing seg2grasp (editable, no deps)"
uv pip install -e "$REPO" --no-deps

cat <<'EOF'
>>> Done. Qwen env ready at venvs/qwen

The Qwen weights download from Hugging Face to the standard HF cache
(~/.cache/huggingface) on first use — nothing to link. Set SEG2GRASP_QWEN_REPO to
a smaller model id, or to a local directory to use a local copy.

Run the classifier as a service:  source venvs/qwen/bin/activate && python scripts/serve_qwen.py
EOF
