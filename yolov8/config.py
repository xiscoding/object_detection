from pathlib import Path
from openvino.runtime import Core
# System setup
DEVICE = 'GPU'  # Options: 'CPU', 'GPU', 'AUTO'
CORE = Core()
# Paths
IMAGE_PATH = Path('yolov8/data/coco_bike.jpg')
MODELS_DIR = Path('yolov8/models')
OUT_DIR = Path('yolov8/datasets')
CFG_PATH = OUT_DIR / "coco.yaml"

# URLs
DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml"

# Model names
DET_MODEL_NAME = "yolov8n"
SEG_MODEL_NAME = "yolov8n-seg"