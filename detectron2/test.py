from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer_original import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
import os
import math

register_coco_instances("train", {}, "data/train/_annotations.coco.json", "./data/train")
register_coco_instances("valid", {}, "data/valid/_annotations.coco.json", "./data/images")
register_coco_instances("test", {}, "data/test/_annotations.coco.json", "./data/test")

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("valid", )   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 52  # 1 classes (person)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

predictor = DefaultPredictor(cfg)


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, image = capture.read()
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    bboxes = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
    masks = outputs["instances"].to("cpu").pred_masks.numpy()
    img = v.get_image()[:, :, ::-1]
    img = cv2.resize(img, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Custom image', img)
    key = cv2.waitKey(1)
