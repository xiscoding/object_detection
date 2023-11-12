from pathlib import Path
import nncf  # noqa: F811
from ultralytics import YOLO
from openvino.runtime import Core, Model, serialize
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg

class ModelManager:
    def __init__(self, core: Core, device: str, models_dir: Path, det_model_name: str, seg_model_name: str, cfg_path: Path):
        self.core = core
        self.device = device
        self.models_dir = models_dir
        self.det_model_name = det_model_name
        self.seg_model_name = seg_model_name
        self.cfg_path = cfg_path

        self.det_model = None
        self.seg_model = None
        self.quantized_models = {}

    def load_yolo_model(self, model_name: str) -> YOLO:
        model_path = self.models_dir / f'{model_name}.pt'
        model = YOLO(model_path)
        return model, model_path, model.model.names

    def get_det_model(self):
        if self.det_model is None:
            self.det_model = self.load_yolo_model(self.det_model_name)
        return self.det_model

    def get_seg_model(self):
        if self.seg_model is None:
            self.seg_model = self.load_yolo_model(self.seg_model_name)
        return self.seg_model

    def quantize_model(self, model_name: str, openvino_model: Model, data_loader, transform_fn, ignored_scope):
        quantization_dataset = nncf.Dataset(data_loader, transform_fn)
        quantized_model = nncf.quantize(
            openvino_model,
            quantization_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope
        )
        quantized_model_path = self.models_dir / f'{model_name}_openvino_int8_model/{model_name}.xml'
        serialize(quantized_model, str(quantized_model_path))
        return quantized_model, quantized_model_path

    def get_quantized_model(self, model_name: str, openvino_model: Model, data_loader, transform_fn, ignored_scope):
        if model_name not in self.quantized_models:
            self.quantized_models[model_name] = self.quantize_model(model_name, openvino_model, data_loader, transform_fn, ignored_scope)
        return self.quantized_models[model_name]

# Usage
core = Core()
model_manager = ModelManager(core, "CPU", Path('./models'), "yolov8n", "yolov8n-seg", Path('./cfg/coco.yaml'))

# Loading and getting the object detection model
det_model, det_model_path, det_label_map = model_manager.get_det_model()

# For quantization, additional parameters like data_loader, transform_fn, and ignored_scope need to be provided
# Example (pseudo-code, needs actual implementation details):
# quantized_det_model, quantized_det_model_path = model_manager.get_quantized_model("yolov8n", openvino_det_model, data_loader, transform_fn, ignored_scope)
