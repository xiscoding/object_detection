import nncf
import openvino.runtime as ov
import torch
from torchvision import datasets, transforms
from pathlib import Path

model_path = Path("/home/xdoestech/Desktop/object_detection/runs/detect/train24/weights/best_openvino_model/best.xml")
# Instantiate your uncompressed model
model = ov.Core().read_model(model_path)
val_path = Path('/home/xdoestech/Desktop/object_detection/Traffic-Signs-2-4')

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder(val_path, transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)

from openvino.runtime import serialize
models_dir = Path('/home/xdoestech/Desktop/object_detection/yolov8/models/openvino_quantized')
DET_MODEL_NAME = Path('yolov8Signs')
int8_model_det_path = models_dir / f'{DET_MODEL_NAME}_openvino_int8_model/{DET_MODEL_NAME}.xml'
print(f"Quantized detection model will be saved to {int8_model_det_path}")
serialize(quantized_model, str(int8_model_det_path))