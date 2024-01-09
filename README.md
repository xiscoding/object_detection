# object_detection
This is a collection of all of the openVino, roboflow, ultralytics tools.
It is not complete and will be added to.
## DEPENDENCY ISSUES:
    - autodistill_all code has had issues newest version of ultralytics 
        - autodistill uses ultralytics version 8.0.81
        - yolov8 uses ultralytics version 8.0.238 
    - Recommended to create seprate enviroments for autodistill and ultralytics 
    - utils/scrape_yandeximages.py: may need to tweak elements depending on browser and location.

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
- scrape_googleimages.py: uses Selenium to scrape google images
- scrape_yandeximages.py: uses Selenium to scrape yandex images (yandex has provided the most relevant images results imo)

#### yolov8
- This directory contains all files related to yolov8 specifically
- config.py
    - sets up paths and configurations
    - NOT REQUIRED MAY BE USEFUL IF YOU WANT A SINGLE PLACE TO DEFINE SUCH THINGS
- initial_setup.py 
    - downloads coco trained yolov8 model
    - ensures all packages and requirements are functional
- model_benchmark.py
    - tests model performace 
    - compares model stats for different model formats 
    - run after training to decide which model format you want
- model_eval.py
    - evaluates single model on specified dataset
    - run after training to verify model performance matches training stats
    - ideally use new dataset
- model_quantize_nncf
    - creates quantized int8 model in openvino format (.bin, .xml)
    - significantly reduces model size and slightly reduces inference time
- model_setup.py
    - NOT USED ATM 
- model_train.py
    - trains ultralytics yolov8 model on roboflow dataset
    - functions to download roboflow dataset and to train yolo model on roboflow dataset
- notebook_util.py
    - various functions to aid display and evalation 
    - USED BY initial_setup.py 