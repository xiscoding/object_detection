from ultralytics import YOLO
from ultralytics import settings

"""
SAVES PREDICTIONS TO TEXT FILE FOR EACH IMAGE IN A FOLDER.
DATA_PATH: Specifies the data source for inference. 
Can be an image path, video file, directory, URL, or device ID for live feeds.
"""
DATA_PATH = '/media/xdoestech/VT Storage 1/PERCEPTION_DATA_BAGS/test6_ucb_2023-11-21-16-08-14_imgs-20240324T062438Z-001/test6_ucb_2023-11-21-16-08-14_imgs'
MODEL_PATH = '/home/xdoestech/Desktop/object_detection/runs/detect/primed_ucb_vtti_32424/weights/best.pt'
SAVE_DIR = '/media/xdoestech/VT Storage 1/PERCEPTION_DATA_BAGS/annotated_images'

settings.update({'save_dir': SAVE_DIR})
# Load a pretrained YOLOv8n model
model = YOLO(MODEL_PATH)

# Run inference on DATA_PATH with arguments
model.predict(DATA_PATH, save_txt=True, imgsz=640, conf=0.5)