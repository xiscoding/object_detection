from pathlib import Path
import nncf  # noqa: F811
from typing import Dict
from ultralytics import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg


from config import CORE, DEVICE, IMAGE_PATH, DET_MODEL_NAME, SEG_MODEL_NAME, MODELS_DIR, DATA_URL, LABELS_URL, CFG_PATH, CFG_URL, OUT_DIR
from model_setup import setup_det_model, setup_seg_model

##system setup
core = CORE
device = DEVICE #options=('CPU', 'GPU', 'AUTO')
##paths
IMAGE_PATH = IMAGE_PATH
models_dir = MODELS_DIR
DATA_URL = DATA_URL
LABELS_URL = LABELS_URL
CFG_URL = CFG_URL
OUT_DIR = OUT_DIR
CFG_PATH = CFG_PATH

# object detection model
det_model, det_model_path = setup_det_model()
label_map = det_model.model.names

args = get_cfg(cfg=DEFAULT_CFG)
args.data = str(CFG_PATH)

# instance segmentation model
seg_model, seg_model_path = setup_seg_model()

#OBJECT DETECTION 
from ultralytics.yolo.data.utils import check_det_dataset
    #MODEL /  VALIDATOR / DATALOADER
det_ov_model = core.read_model(det_model_path) #cant access
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device)
det_validator = det_model.ValidatorClass(args=args)
det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

#INSTANCE SEGMENTATION
    #MODEL /  VALIDATOR / DATALOADER
seg_ov_model = core.read_model(seg_model_path)
if device != "CPU":
    seg_ov_model.reshape({0: [1, 3, 640, 640]})
seg_compiled_model = core.compile_model(seg_ov_model, device)
seg_validator = seg_model.ValidatorClass(args=args)
seg_validator.data = check_det_dataset(args.data)
seg_data_loader = seg_validator.get_dataloader("datasets/coco/", 1)

def transform_fn(data_item:Dict):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
       data_item: Dict with data item produced by DataLoader during iteration
    Returns:
        input_tensor: Input data for quantization
    """
    input_tensor = det_validator.preprocess(data_item)['img'].numpy()
    return input_tensor


quantization_dataset = nncf.Dataset(det_data_loader, transform_fn)

'''
nncf.quantize:
interface for model quantization

REQUIREMENTS: instance of the OpenVINO Model and quantization dataset.

Optionally some additional parameters for the configuration quantization process 
(number of samples for quantization, preset, ignored scope, etc.) can be provided
'''

ignored_scope = nncf.IgnoredScope(
    types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
    names=[
        "/model.22/dfl/conv/Conv",           # in the post-processing subgraph
        "/model.22/Add",
        "/model.22/Add_1",
        "/model.22/Add_2",
        "/model.22/Add_3",
        "/model.22/Add_4",
        "/model.22/Add_5",
        "/model.22/Add_6",
        "/model.22/Add_7",
        "/model.22/Add_8",
        "/model.22/Add_9",
        "/model.22/Add_10"
    ]
)

#QUANTIZED MODELS
# Detection model
quantized_det_model = nncf.quantize(
    det_ov_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    ignored_scope=ignored_scope
)
from openvino.runtime import serialize
int8_model_det_path = models_dir / f'{DET_MODEL_NAME}_openvino_int8_model/{DET_MODEL_NAME}.xml'
print(f"Quantized detection model will be saved to {int8_model_det_path}")
serialize(quantized_det_model, str(int8_model_det_path))

# Instance segmentation model
quantized_seg_model = nncf.quantize(
    seg_ov_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    ignored_scope=ignored_scope
)
int8_model_seg_path = models_dir / f'{SEG_MODEL_NAME}_openvino_int8_model/{SEG_MODEL_NAME}.xml'
print(f"Quantized segmentation model will be saved to {int8_model_seg_path}")
serialize(quantized_seg_model, str(int8_model_seg_path))

if __name__ == '__main__':
    import cv2
    from PIL import Image
    import numpy as np
    from initial_setup import draw_results
    from openvino_setup import detect

    #TEST QUANTIZED MODELS
    if device != "CPU":
        quantized_det_model.reshape({0: [1, 3, 640, 640]})
    quantized_det_compiled_model = core.compile_model(quantized_det_model, device)
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