from pathlib import Path
from ultralytics import YOLO
from config import MODELS_DIR, DET_MODEL_NAME, SEG_MODEL_NAME

def setup_model(model_name: str, export_openvino: bool = True):
    """
    Set up a YOLO model.
    
    Parameters:
        model_name (str): Name of the model to set up.
        export_openvino (bool): Whether to export the model to OpenVINO format.

    Returns:
        model: The YOLO model instance.
    """
    model_path = MODELS_DIR / f'{model_name}.pt'

    # Ensure the models directory exists
    MODELS_DIR.mkdir(exist_ok=True)

    # Create the model instance
    model = YOLO(model_path)

    # Export to OpenVINO format if required
    if export_openvino:
        openvino_model_path = MODELS_DIR / f"{model_name}_openvino_model/{model_name}.xml"
        if not openvino_model_path.exists():
            model.export(format="openvino", dynamic=True, half=False)

    return model, openvino_model_path

# Functions to specifically handle object detection and instance segmentation models
def setup_det_model():
    return setup_model(DET_MODEL_NAME)

def setup_seg_model():
    return setup_model(SEG_MODEL_NAME)
