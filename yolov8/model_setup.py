from pathlib import Path
from ultralytics import YOLO
from config import MODELS_DIR, DET_MODEL_NAME, SEG_MODEL_NAME

def setup_model(model_name: str, export_openvino: bool = True, model_path= "/home/xdoestech/Desktop/object_detection/yolov8s.pt"):
    """
    Set up a YOLO model. IN OPENVINO FORMAT
    
    Parameters:
        model_name (str): Name of the model to set up.
        export_openvino (bool): Whether to export the model to OpenVINO format.
        model_path (str): Path to model ***defaults to "/home/xdoestech/Desktop/object_detection/yolov8s.pt"***
    Returns:
        model: The YOLO model instance.
    """
    #model_path = MODELS_DIR / f'{model_name}.pt'
    

    MODELS_DIR = Path("/home/xdoestech/Desktop/object_detection/models/traffic_12_13_23")
    # Ensure the models directory exists
    MODELS_DIR.mkdir(exist_ok=True)

    # Create the model instance
    model = YOLO(model_path)

    # Export to OpenVINO format if required
    ##model.export is responsible for model conversion
    ##conver to openVino IR
    if export_openvino:
        openvino_model_path = MODELS_DIR / f"{model_name}_openvino_model/{model_name}.xml"
        if not openvino_model_path.exists():
            model.export(format="openvino", dynamic=True, half=False)
        return model, openvino_model_path
    else:
        return model

# Functions to specifically handle object detection and instance segmentation models
def setup_det_model(export_openvino, det_model_path):
    return setup_model(DET_MODEL_NAME, export_openvino, det_model_path)

def setup_seg_model():
    return setup_model(SEG_MODEL_NAME)


if __name__ == '__main__':
    model, openvino_model_path = setup_model(DET_MODEL_NAME)
    print(f"Open vino model path: {openvino_model_path}")