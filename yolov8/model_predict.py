from ultralytics import YOLO

ucb = '/home/xdoestech/Desktop/object_detection/runs/detect/primed_ucb_vtti_32424/weights/best.pt'
ucb2 = '/home/xdoestech/Desktop/object_detection/runs/detect/primed_ucb_32424/weights/best.pt'
all_signs = '/home/xdoestech/Desktop/object_detection/runs/detect/train39/weights/best.pt'
MODEL_PATH = ucb2
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

    # Run inference with arguments
    model.predict(PATH, save=True, imgsz=640, conf=0.3)

def predict_video(PATH):
    # Load a pretrained YOLOv8n model
    model = YOLO(all_signs_path)

    # Run inference on 'bus.jpg' with arguments
    model.predict(PATH, show=True)


import cv2
from ultralytics import YOLO
def predict_video_fancy(PATH):

    model = YOLO(all_signs_path)
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
    IMG_PATH = '/home/xdoestech/Desktop/object_detection/testImage_ucbStopline.png'
    predict_specialSettings(IMG_PATH)
    