from ultralytics import YOLO
from roboflow import Roboflow
import os
from keys import roboflow_key_school, roboflow_key_business
"""
DOWNLOAD DATASET OFF ROBOFLOW
"""
def download_dataset_roboflow():
    rf = Roboflow(api_key=roboflow_key_business)
    # project = rf.workspace("autodrive-a25la").project("traffic-signs-8vjvi")
    # dataset = project.version(7).download("yolov8")
    # project = rf.workspace("autodrive-a25la").project("traffic-signs-2-ny3ak")
    # dataset = project.version(7).download("yolov8")
    # project = rf.workspace("autodrive-a25la").project("spdlimit_yield_stopsign")
    # dataset = project.version(2).download("yolov8")
    # project = rf.workspace("autodrive-a25la").project("spdlimit_yield_stopsign_barrel_pedestrian")
    # dataset = project.version(3).download("yolov8")
    # project = rf.workspace("autodrive-a25la").project("trafficsign_modelprimer")
    # dataset = project.version(2).download("yolov8")
    # project = rf.workspace("trafficdata-xxj8s").project("trafficdata_v2")
    # version = project.version(1)
    # dataset = version.download("yolov8")
    # project = rf.workspace("trafficdata-xxj8s").project("primer_v1") 
    # version = project.version(2)
    # dataset = version.download("yolov8")
    project = rf.workspace("trafficdata-xxj8s").project("trafficdata_v2") #BUSINESS ROBOFLOW
    version = project.version(2) 
    dataset = version.download("yolov8")


"""
TRAIN MODEL
saves to runs/detect/train{RUN NUMBER}/weights
"""
def train_yolov8():
    #SET PATHS
    data_loc = '/home/xdoestech/Desktop/object_detection/TrafficData_v2-2'
    model_loc = 'runs/detect/primed_yolov8s/weights/best.pt'
    model = YOLO(model_loc)  # load a pretrained model (recommended for training)
    results = model.train(data=f'{data_loc}/data.yaml', epochs=300, imgsz=640, patience=100)

# os.system(f'yolo task=detect mode=train model=yolov8s.pt data={data_loc}/data.yaml epochs=25 imgsz=640 plots=True')
if __name__ == '__main__':
    #download_dataset_roboflow()
    train_yolov8()
