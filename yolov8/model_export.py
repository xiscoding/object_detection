from pathlib import Path
from openvino.runtime import Core
from ultralytics import YOLO
import os
#       System setup
DEVICE = 'GPU'  # Options: 'CPU', 'GPU', 'AUTO'
CORE = Core()
MODELS_DIR = '/home/xdoestech/Desktop/object_detection/yolov8/models'
#       SET PATHS
ultralytics_model_path = '/home/xdoestech/Desktop/object_detection/runs/detect/train24/weights/best.pt'
ultralytics_model = YOLO(ultralytics_model_path)
model_name = 'yolov8Signs'


def ultralytics_to_openvino():
    """
    model export format: https://docs.ultralytics.com/modes/export/#usage-examples
    openvino model saved in same directory as ultralytics model
    """
    # openvino_model_path = Path(os.path.join(MODELS_DIR, f"{model_name}_openvino_model/{model_name}.xml"))
    # if not openvino_model_path.exists():
    ultralytics_model.export(format="openvino", dynamic=True, half=False)
    # return openvino_model_path
    # else:
    #     return f"model already exists at {openvino_model_path}"


if __name__ == '__main__':
    ultralytics_to_openvino()
