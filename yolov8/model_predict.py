from ultralytics import YOLO

MODEL_PATH = '/home/xdoestech/Desktop/object_detection/runs/detect/train39/weights/best.pt'
IMG_PATH = '/home/xdoestech/Desktop/object_detection/testImage2.jpg'
VID_PATH = '/home/xdoestech/Desktop/object_detection/vtti_12924_run1.mp4'
def predict_directory(DIR_PATH):
    # Load a pretrained YOLOv8n model
    #model = YOLO('yolov8n.pt')
    model = YOLO(MODEL_PATH)
    # Define path to directory containing images and videos for inference
    source = DIR_PATH

    # Run inference on the source
    results = model(source, stream=True)  # generator of Results objects

def predict_singleImage(IMG_PATH):
    # Load a pretrained YOLOv8n model
    model = YOLO(MODEL_PATH)

    # Define path to the image file
    source = IMG_PATH

    # Run inference on the source
    results = model(source)  # list of Results objects

def predict_specialSettings(PATH):
    model = YOLO(MODEL_PATH)

    # Run inference on 'bus.jpg' with arguments
    model.predict(PATH, save=True, imgsz=320, conf=0.5)

def predict_video(PATH):
    # Load a pretrained YOLOv8n model
    model = YOLO(MODEL_PATH)

    # Run inference on 'bus.jpg' with arguments
    model.predict(PATH, show=True)
if __name__ == '__main__':
    IMG_PATH = ''
    predict_video(VID_PATH)
    