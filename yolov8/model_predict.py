from ultralytics import YOLO

primed_intersection = '/home/xdoestech/Desktop/object_detection/runs/detect/primed_ucb_32424/weights/best.pt'
vtti = '/home/xdoestech/Desktop/object_detection/runs/detect/train39/weights/best.pt'
competition_signs_original = '/home/xdoestech/Desktop/object_detection/runs/detect/competition_signs_1/weights/best.pt'
competition_signs = '/home/xdoestech/Desktop/object_detection/yolov8/runs_4_9_24/run1_dfl_5_coslr_True/weights/best.pt'
MODEL_PATH = competition_signs
all_signs_path = '/home/xdoestech/Desktop/object_detection/runs/detect/train33/weights/best.pt'
IMG_PATH = '/home/xdoestech/Desktop/object_detection/testImage_ucbStopline.png'
VID_PATH = '/home/xdoestech/Desktop/object_detection/ucb_2023_11_21_test5.mp4'
def predict_directory(DIR_PATH):
    # Load a pretrained YOLOv8n model
    #model = YOLO('yolov8n.pt')
    model = YOLO(MODEL_PATH)
    # Define path to directory containing images and videos for inference
    source = DIR_PATH

    # Run inference on the source
    results = model(source, stream=True, save_txt=True)  # generator of Results objects

def predict_singleImage(IMG_PATH):
    # Load a pretrained YOLOv8n model
    model = YOLO(MODEL_PATH)

    # Define path to the image file
    source = IMG_PATH

    # Run inference on the source
    results = model(source)  # list of Results objects

def predict_specialSettings(PATH):
    model = YOLO(MODEL_PATH)

    # Run inference with arguments
    model.predict(PATH, save=True, imgsz=640, conf=0.5)

def predict_video(PATH):
    # Load a pretrained YOLOv8n model
    model = YOLO(MODEL_PATH)

    # Run inference on 'bus.jpg' with arguments
    model.predict(PATH, show=True, save_frames=True, save=True)


import cv2
from ultralytics import YOLO
def predict_video_fancy(PATH):

    model = YOLO(MODEL_PATH)
    # Open the video file
    video_path = PATH
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = '/home/xdoestech/Desktop/object_detection/TrafficData_v3-2/valid/images/20231121_134826_jpg.rf.73f67d54975af127bc1488b63728d404.jpg'
    IMG_PATH = '/home/xdoestech/Desktop/object_detection/testImage_ucbStopline.png'
    dir_33024 = '/home/xdoestech/Desktop/object_detection/test_image_33024'
    #predict_video('/home/xdoestech/Desktop/object_detection/ucb_2023_11_21_test5.mp4')
    predict_specialSettings(dir_33024)
    