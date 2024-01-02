import torch, torchvision
import os, json, cv2, random
import requests

# Import Detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

# Setup logger for Detectron2
setup_logger()

# Function to download an image using requests
def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)

# Download an example image
image_url = "http://images.cocodataset.org/val2017/000000439715.jpg"
image_path = "input.jpg"
download_image(image_url, image_path)

# Read the image using OpenCV
im = cv2.imread(image_path)
if im is None:
    raise Exception("Image not found")

# Configuration for the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'

# Create predictor and make prediction
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Output classes and bounding boxes
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# Visualize the predictions on the image
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
