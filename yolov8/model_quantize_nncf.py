import nncf
import openvino.runtime as ov
import torch
from torchvision import datasets, transforms
from pathlib import Path
"""
source: https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/230-yolov8-optimization/230-yolov8-object-detection.ipynb#scrollTo=b2e8cec4-d0b3-4da0-b54c-7964e7bcbfe2
source: https://github.com/openvinotoolkit/nncf?tab=readme-ov-file
"""
model_path = Path("/home/xdoestech/Desktop/object_detection/runs/detect/train24/weights/best_openvino_model/best.xml")
# Instantiate your uncompressed model
det_ov_model = ov.Core().read_model(model_path)
val_path = Path('/home/xdoestech/Desktop/object_detection/Traffic-Signs-2-4')

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder(val_path, transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
quantization_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
# quantized_model = nncf.quantize(model, calibration_dataset)
"""
YOLOv8 model contains non-ReLU activation functions, which require asymmetric quantization of activations. 
To achieve a better result, we will use a mixed quantization preset. 
It provides symmetric quantization of weights and asymmetric quantization of activations. 
For more accurate results, we should keep the operation in the postprocessing subgraph in floating point precision, using the ignored_scope parameter.
"""
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

# Detection model
quantized_det_model = nncf.quantize(
    det_ov_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    ignored_scope=ignored_scope
)
from openvino.runtime import serialize
models_dir = Path('/home/xdoestech/Desktop/object_detection/yolov8/models/openvino_quantized')
DET_MODEL_NAME = Path('yolov8Signs_1')
int8_model_det_path = models_dir / f'{DET_MODEL_NAME}_openvino_int8_model/{DET_MODEL_NAME}.xml'
print(f"Quantized detection model will be saved to {int8_model_det_path}")
serialize(quantized_det_model, str(int8_model_det_path))