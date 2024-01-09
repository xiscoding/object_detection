from pathlib import Path
from typing import Tuple
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.utils import ops
import torch
import numpy as np
from openvino.runtime import Core, Model
from ultralytics.yolo.engine.validator import BaseValidator as Validator

from notebook_utils import draw_results
from openvino_utils import image_to_tensor, postprocess, preprocess_image
from config import DET_MODEL_NAME, SEG_MODEL_NAME, CORE, DEVICE, IMAGE_PATH, MODELS_DIR
"""
Uses OpenVINO IR model. Model creation in model_setup

If new or different model run model_setup.py
"""

##system setup
core = CORE
device = DEVICE #options=('CPU', 'GPU', 'AUTO')
##paths
IMAGE_PATH = IMAGE_PATH
models_dir = MODELS_DIR

##model setup
# object detection model
models_dir.mkdir(exist_ok=True)
DET_MODEL_NAME = DET_MODEL_NAME
det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names
##det_model_path -> model_setup.py
det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)

# instance segmentation model
SEG_MODEL_NAME = SEG_MODEL_NAME

seg_model = YOLO(models_dir / f'{SEG_MODEL_NAME}.pt')
seg_model_path = models_dir / f"{SEG_MODEL_NAME}_openvino_model/{SEG_MODEL_NAME}.xml"
if not seg_model_path.exists():
    seg_model.export(format="openvino", dynamic=True, half=False)

## OPEN VINO MODELS
#OBJECT DETECTION
det_ov_model = core.read_model(det_model_path)
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device)

#INSTANCE SEGMENTATION
seg_ov_model = core.read_model(seg_model_path)
if device != "CPU":
    seg_ov_model.reshape({0: [1, 3, 640, 640]})
seg_compiled_model = core.compile_model(seg_ov_model, device)

def detect(image:np.ndarray, model:Model):
    """
    OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
    Parameters:
        image (np.ndarray): input image.
        model (Model): OpenVINO compiled model.
    Returns:
        detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
    """
    num_outputs = len(model.outputs)
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    masks = None
    if num_outputs > 1:
        masks = result[model.output(1)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
    return detections

###RUN OBJECT AND SEGMENTATION INFERENCE
if __name__ == '__main__':
    #RUN OBJECT DETECTION
    input_image = np.array(Image.open(IMAGE_PATH))
    detections = detect(input_image, det_compiled_model)[0]
    image_with_boxes = draw_results(detections, input_image, label_map) 

    # Convert from RGB (PIL) to BGR (OpenCV)
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imshow("Object Detection", image_with_boxes)
    cv2.waitKey(0)

    #RUN INSTANCE SEGMENTATION

    input_image = np.array(Image.open(IMAGE_PATH))
    detections = detect(input_image, seg_compiled_model)[0]
    image_with_masks = draw_results(detections, input_image, label_map)

    # Convert from RGB (PIL) to BGR (OpenCV)
    image_with_masks = cv2.cvtColor(image_with_masks, cv2.COLOR_RGB2BGR)
    cv2.imshow("Instance Segmentation", image_with_masks)
    cv2.waitKey(0)

    # When all processing is done, destroy the windows
    cv2.destroyAllWindows()