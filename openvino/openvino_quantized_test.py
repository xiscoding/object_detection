import numpy as np
from openvino.runtime import Core, Type, Layout, Model, serialize
from openvino.preprocess import PrePostProcessor
from pathlib import Path
from config import CFG_PATH, DEVICE, IMAGE_PATH, DET_MODEL_NAME
from notebook_utils import draw_results
from openvino_utils import letterbox, postprocess
from PIL import Image
from ultralytics.yolo.engine.validator import BaseValidator as Validator
"""
Preform PrePostProcessing
"""
def setup_preprocessing(model, int8_model_det_path):
    ppp = PrePostProcessor(model)
    ppp.input(0).tensor().set_shape([1, 640, 640, 3]).set_element_type(Type.u8).set_layout(Layout('NHWC'))
    ppp.input(0).preprocess().convert_element_type(Type.f32).convert_layout(Layout('NCHW')).scale([255., 255., 255.])

    # Build the model with preprocessing and save to OpenVINO IR format
    quantized_model_with_preprocess = ppp.build()
    serialize(quantized_model_with_preprocess, str(int8_model_det_path.with_name(f"{DET_MODEL_NAME}_with_preprocess.xml")))
    return quantized_model_with_preprocess

def detect_without_preprocess(image: np.ndarray, model: Model):
    """
    Perform inference using an OpenVINO model with integrated preprocessing.
    
    Parameters:
        image (np.ndarray): The input image.
        model (Model): The OpenVINO compiled model.

    Returns:
        np.ndarray: Detected boxes in the format [x1, y1, x2, y2, score, label].
    """
    output_layer = model.output(0)
    img = letterbox(image)[0]
    input_tensor = np.expand_dims(img, 0)
    input_hw = img.shape[:2]
    result = model(input_tensor)[output_layer]
    detections = postprocess(result, input_hw, image)
    return detections

if __name__ == '__main__':
    from ultralytics.yolo.cfg import get_cfg
    from model_setup import setup_det_model, setup_seg_model
    from ultralytics.yolo.utils import DEFAULT_CFG
    from ultralytics.yolo.cfg import get_cfg
    import nncf
    from openvino_quantize_nncf import ignored_scope, transform_fn
    det_model, det_model_path = setup_det_model()
    label_map = det_model.model.names
    device = DEVICE
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(CFG_PATH)
    core = Core()
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
    quantization_dataset = nncf.Dataset(det_data_loader, transform_fn)
    #QUANTIZED MODELS
    # Detection model
    quantized_det_model = nncf.quantize(
        det_ov_model,
        quantization_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope
    )
    quantized_model_with_preprocess = setup_preprocessing(quantized_det_model, Path(IMAGE_PATH))
    compiled_model = core.compile_model(quantized_model_with_preprocess, DEVICE)

    input_image = np.array(Image.open(IMAGE_PATH))
    detections = detect_without_preprocess(input_image, compiled_model)[0]
    image_with_boxes = draw_results(detections, input_image, label_map)

    # Handling the image display or processing can be done here or in the calling code.
