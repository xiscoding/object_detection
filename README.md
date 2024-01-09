# object_detection
This is a collection of all of the openVino, roboflow, ultralytics tools.
It is not complete and will be added to.

## Directories: 
#### autodistill_all: 
- This directory contains scripts and utilities for running AutoDistill
- autodistill_utils: 
    - display_utils.py: utility to display graphs in python
    - vid_to_img.py: convert video to img frames
- run_autodistill.py: 
    - create distilled models with DINO or other LFMs 
    - helps with autolabeling in theory

#### utils
- This directory contains general utility files that are not specific to any system
- scrape_googleimages.py: uses Selenium to scape google images

#### yolov8
- This directory contains all files related to yolov8 specifically
- config.py
    - sets up paths and configurations
- model_manager.py
    - NOT FUNCTIONAL OR IN USE CURRENTLY
- model_setup.py
    - creates standard yolov8(ultralytics) and openVino models from path
- notebook_utils.py
    - utilties specific to yolov8(ultralytics) and openVino model creation and quantization
    - includes some display helpers and other helpful utilities
- openvino_utils
    - openVINO utilities 
    - letterbox, preprocess, image_to_tensor, postprocess
- objectdetect_livedemo.py
    - a demonstration of openVino with live video footage or images
- openvino_compare_model_performance.py
    - compares performance between two models (uses benchmark_app)
    - outputs results to console
- openvino_preprocessing.py
    -