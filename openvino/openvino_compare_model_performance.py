'''
https://docs.openvino.ai/2023.1/notebooks/230-yolov8-optimization-with-output.html
'''
from pathlib import Path
from openvino.runtime import Core
from ultralytics import YOLO
import os
from openvino_quantize_nncf import transform_fn

from openvino_validation import print_stats, test
from config import DET_MODEL_NAME, SEG_MODEL_NAME, CORE, DEVICE, IMAGE_PATH, MODELS_DIR, DATA_URL, LABELS_URL, CFG_PATH, CFG_URL, OUT_DIR

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

DET_MODEL_NAME = DET_MODEL_NAME
SEG_MODEL_NAME = SEG_MODEL_NAME

det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
int8_model_det_path = models_dir / f'{DET_MODEL_NAME}_openvino_int8_model/{DET_MODEL_NAME}.xml'

seg_model_path = models_dir / f"{SEG_MODEL_NAME}_openvino_model/{SEG_MODEL_NAME}.xml"
int8_model_seg_path = models_dir / f'{SEG_MODEL_NAME}_openvino_int8_model/{SEG_MODEL_NAME}.xml'

'''
For more accurate performance, it is recommended to run benchmark_app in a 
terminal/command prompt after closing other applications. 
Run benchmark_app -m <model_path> -d CPU -shape "<input_shape>" 
to benchmark async inference on CPU on specific input data shape for one minute. 
Change CPU to GPU to benchmark on GPU. 
Run benchmark_app --help to see an overview of all command-line options.
'''
##Compare performance object detection models
#OBJECT DETECTION MODELS 32bit, 8bit
# Inference FP32 model (OpenVINO IR)
print("!!!!!!!!!!!FP32 OBJECT DETECTION MODEL!!!!!!!!")
os.system(f'benchmark_app -m {det_model_path} -d {device} -api async -shape "[1,3,640,640]"')
# Inference INT8 model (OpenVINO IR)
print("!!!!!!!!!!!INT8 OBJECT DETECTION MODEL!!!!!!!!")
os.system(f'benchmark_app -m {int8_model_det_path} -d {device} -api async -shape "[1,3,640,640]"')

#INSTANCE SEGMENTATION MODELS 32bit, 8bit
# Inference FP32 model (OpenVINO IR)
#os.system(f'benchmark_app -m {det_model_path} -d {device} -api async -shape "[1,3,640,640]"')
# Inference INT8 model (OpenVINO IR)
#os.system(f'benchmark_app -m {int8_model_det_path} -d {device} -api async -shape "[1,3,640,640]"')

#QUANTIZED MODELS
import nncf  # noqa: F811
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
#OBJECT DETECTION 
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import ops
from model_setup import setup_det_model, setup_seg_model

NUM_TEST_SAMPLES = 300

#MODEL /  VALIDATOR / DATALOADER
det_model, det_ov_model_path = setup_det_model()
label_map = det_model.names

####DET VALIDATOR#######
args = get_cfg(cfg=DEFAULT_CFG)
args.data = str(CFG_PATH)
det_ov_model = core.read_model(det_model_path) #cant access

det_validator = det_model.ValidatorClass(args=args)
det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

det_validator.is_coco = True
det_validator.class_map = ops.coco80_to_coco91_class()
det_validator.names = det_model.model.names
det_validator.metrics.names = det_validator.names
det_validator.nc = det_model.model.model[-1].nc
####DET VALIDATOR#######

if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
quantization_dataset = nncf.Dataset(det_data_loader, transform_fn)


# Detection model
quantized_det_model = nncf.quantize(
    det_ov_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    ignored_scope=ignored_scope
)
fp_det_stats = test(det_ov_model, core, det_data_loader, det_validator, num_samples=NUM_TEST_SAMPLES)
int8_det_stats = test(quantized_det_model, core, det_data_loader, det_validator, num_samples=NUM_TEST_SAMPLES)
print("!!!!!!!!!!!FP32 model accuracy!!!!!!!!!")
print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())

print("!!!!!!!!!!!INT8 model accuracy!!!!!!!!!!")
print_stats(int8_det_stats, det_validator.seen, det_validator.nt_per_class.sum())