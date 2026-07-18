"""Central path resolution for Seg2Grasp.

All paths resolve relative to the repo root by default, but every root can be
overridden with an environment variable so the code stays portable across machines:

    SEG2GRASP_DATA        -> data dir            (default: <repo>/data)
    SEG2GRASP_DATASETS    -> datasets            (default: <DATA>/datasets)
    SEG2GRASP_CHECKPOINTS -> model weights       (default: <DATA>/checkpoints)
    SEG2GRASP_OUTPUTS     -> run outputs / viz   (default: <DATA>/outputs)

`<repo>/data` is expected to be a symlink to a large storage volume (heavy
checkpoints, datasets, and outputs live there, not in the git tree).
"""
import os

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))   # <repo>/seg2grasp
REPO_ROOT = os.path.dirname(PKG_ROOT)                    # <repo>

DATA_ROOT = os.environ.get("SEG2GRASP_DATA", os.path.join(REPO_ROOT, "data"))
DATASET_ROOT = os.environ.get("SEG2GRASP_DATASETS", os.path.join(DATA_ROOT, "datasets"))
CHECKPOINT_ROOT = os.environ.get("SEG2GRASP_CHECKPOINTS", os.path.join(DATA_ROOT, "checkpoints"))
OUTPUT_ROOT = os.environ.get("SEG2GRASP_OUTPUTS", os.path.join(DATA_ROOT, "outputs"))

CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")

# --- Segmentation (Mask2Former) -------------------------------------------------
# The trained checkpoint dir holds `model_final.pth` + `config.yaml` together
# (the DA-Fusion legacy best RGB-D run: DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05).
SEG_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "segmentation")
SEG_WEIGHTS = os.path.join(SEG_CHECKPOINT_DIR, "model_final.pth")
SEG_CONFIG = os.path.join(SEG_CHECKPOINT_DIR, "config.yaml")

# --- Classification (Qwen VLM) --------------------------------------------------
# Selectable model: pass a short alias (below), a full HF repo id, or a local
# directory path via --model / SEG2GRASP_QWEN_REPO. Weights download to the
# HuggingFace cache (~/.cache/huggingface) on first use — no local link needed.
QWEN_MODELS = {
    "35b": "Qwen/Qwen3.6-35B-A3B",       # bf16 — default (reliable)
    "27b-fp8": "Qwen/Qwen3.6-27B-FP8",   # smaller FP8; needs a working FP8 stack
                                         # (broken/slow on NVIDIA SM120 / Blackwell here)
}
QWEN_DEFAULT_MODEL = os.environ.get("SEG2GRASP_QWEN_REPO", "35b")


def resolve_qwen_model(name=None):
    """Resolve a Qwen model selection to a HF repo id or local path.

    ``name`` may be an alias in ``QWEN_MODELS``, a full HF repo id, or a local
    directory path; unknown values pass through unchanged. ``None`` uses the
    default (``QWEN_DEFAULT_MODEL``).
    """
    name = name or QWEN_DEFAULT_MODEL
    return QWEN_MODELS.get(name, name)

# --- Category list for classification ------------------------------------------
CATEGORIES_CONFIG = os.path.join(CONFIG_ROOT, "categories.yaml")
