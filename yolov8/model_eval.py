from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/home/xdoestech/Desktop/object_detection/runs/detect/train22/weights/best.pt')  # load a custom model

data_loc = '/home/xdoestech/Desktop/object_detection/Traffic-Signs-2-4'

# Validate the model
metrics = model.val(data=f'{data_loc}/data.yaml')  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category