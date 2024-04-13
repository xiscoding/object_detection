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
    # project = rf.workspace("trafficdata-xxj8s").project("trafficdata_v2") #BUSINESS ROBOFLOW
    # version = project.version(4)) 
    # dataset = version.download("yolov8")
    # project = rf.workspace("trafficdata-xxj8s").project("trafficdata_v3")
    # version = project.version(1)
    # dataset = version.download("yolov8")
    project = rf.workspace("trafficdata-xxj8s").project("trafficdata_v3")
    version = project.version(4)
    dataset = version.download("yolov8")



"""
TRAIN MODEL
saves to runs/detect/train{RUN NUMBER}/weights

docs: https://docs.ultralytics.com/modes/train/#train-settings
loss functions: https://github.com/ultralytics/ultralytics/issues/4025
dfl_loss: https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/#aioseo-distributed-focal-loss (possible error explaining dirac delta distribution)
label smoothing: https://towardsdatascience.com/label-smoothing-make-your-model-less-over-confident-b12ea6f81a9a
cos_lr: adjusting the learning rate following a cosine curve over epochs
"""
def train_yolov8():
    #SET PATHS
    data_loc = '/home/xdoestech/Desktop/object_detection/TrafficData_v3-4'
    model_loc = '/home/xdoestech/Desktop/object_detection/yolov8/runs_3_30_24/run3_dfl_5_coslr_True_lrf_00001/weights/best.pt'
    model = YOLO(model_loc)  # load a pretrained model (recommended for training)
    results = model.train(
        data=f'{data_loc}/data.yaml', 
        epochs=400,
        time = None, 
        imgsz=640, 
        patience=200,
        box = 7.5,
        dfl = 5,
        cls = 0.5,
        cos_lr =True,
        batch = -1,
        dropout = 0.0,
        project = '/home/xdoestech/Desktop/object_detection/yolov8/runs_4_9_24',
        name = 'run1_dfl_5_coslr_True',
        plots = True
        )

# os.system(f'yolo task=detect mode=train model=yolov8s.pt data={data_loc}/data.yaml epochs=25 imgsz=640 plots=True')
if __name__ == '__main__':
    #download_dataset_roboflow()
    train_yolov8()
