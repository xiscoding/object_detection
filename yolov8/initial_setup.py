##TEST SYSTEM IS SET UP

##Download yolov8 dectection and segmentation models
###Run inference on test image
import os
from pathlib import Path
import random
import ultralytics
from ultralytics import YOLO
from notebook_utils import download_file, VideoPlayer

from typing import Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from ultralytics.yolo.utils.plotting import colors
from notebook_utils import download_file
from model_setup import setup_det_model, setup_seg_model

def download_test_sample():
    # Download a test sample

    download_file(
        url='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg',
        filename=IMAGE_PATH.name,
        directory=IMAGE_PATH.parent
    )

"""
print results from test sample
"""
def test_sample_det(det_model_path):
    """
    set up new yolo model 
    """
    det_model = setup_det_model(False, det_model_path)

    #IF OPEN VINO
    #det_model, det_ov_model_path = setup_det_model(False, det_model_path)

    res = det_model(IMAGE_PATH)
    pil_image = Image.fromarray(res[0].plot()[:, :, ::-1])  # Convert image to PIL format
    pil_image.show()  # Display the image

def test_sample_seg():

    seg_model, seg_ov_model_path = setup_seg_model()
    res = seg_model(IMAGE_PATH)
    pil_image = Image.fromarray(res[0].plot()[:, :, ::-1])  # Convert image to PIL format
    pil_image.show()  # Display the image

if __name__ == '__main__':
    ultralytics.checks()
    #os.system('unset GTK_PATH')
    #set up model paths
    IMAGE_PATH = Path('yolov8/data/coco_bike.jpg')
    models_dir = Path('yolov8/models')
    model_path = Path('/home/xdoestech/Desktop/object_detection/yolov8s.pt')

    test_sample_det(model_path)
    #test_sample_seg()
