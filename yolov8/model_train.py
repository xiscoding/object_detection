from ultralytics import YOLO
from roboflow import Roboflow
import os

"""
DOWNLOAD DATASET OFF ROBOFLOW
"""
def download_dataset_roboflow():
    rf = Roboflow(api_key="M94jpLaAoSmGUKIvl711")
    # project = rf.workspace("autodrive-a25la").project("traffic-signs-8vjvi")
    # dataset = project.version(7).download("yolov8")
    project = rf.workspace("autodrive-a25la").project("traffic-signs-2-ny3ak")
    dataset = project.version(4).download("yolov8")

"""
TRAIN MODEL
saves to runs/detect/train{RUN NUMBER}/weights
"""
def train_yolov8():
    #SET PATHS
    data_loc = '/home/xdoestech/Desktop/object_detection/Traffic-Signs-2-4'
    model_loc = '/home/xdoestech/Desktop/object_detection/runs/detect/train22/weights/best.pt'
    model = YOLO(model_loc)  # load a pretrained model (recommended for training)
    results = model.train(data=f'{data_loc}/data.yaml', epochs=25, imgsz=640)

# os.system(f'yolo task=detect mode=train model=yolov8s.pt data={data_loc}/data.yaml epochs=25 imgsz=640 plots=True')
if __name__ == '__main__':
    #download_dataset_roboflow()
    train_yolov8()