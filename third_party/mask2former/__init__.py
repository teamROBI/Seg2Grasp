# Copyright (c) Facebook, Inc. and its affiliates.
# (Vendored for Seg2Grasp; __init__ split so inference works without the
#  training-only dataset registration, which needs dataset paths.)

# --- inference essentials (registers backbones, pixel decoders, decoders, meta-arch) ---
from . import modeling
from .config import add_maskformer2_config
from .maskformer_model import MaskFormer

# --- training-only components -------------------------------------------------
# Dataset registration, dataset mappers, TTA and evaluation are only needed for
# (re)training. They require dataset paths (a `legacy_paths` module) that are not
# part of the inference release, so they are imported best-effort.
try:
    from . import data  # register all datasets
    from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
    from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
    from .data.dataset_mappers.mask_former_instance_dataset_mapper import MaskFormerInstanceDatasetMapper
    from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import MaskFormerPanopticDatasetMapper
    from .data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
    from .data.dataset_mappers.uoais_dataset_mapper import DatasetMapperWithBasis
    from .test_time_augmentation import SemanticSegmentorWithTTA
    from .evaluation.instance_evaluation import InstanceSegEvaluator
except Exception:  # noqa: BLE001 — training extras are optional at inference time
    pass
