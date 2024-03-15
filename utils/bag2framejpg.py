import rosbags
from pathlib import Path
import cv2
import os

# Define the path to the bag file and the output folder
bag_path = Path('/media/xdoestech/VT Storage 1/Bag/raw_bagfiles/vtti_1_29_24_run_1')
output_folder = Path('/media/xdoestech/VT Storage 1/Bag/extracted_bagfiles/vtti_1_29_24_run_1')

# Check if the output folder exists, if not, create it
if not output_folder.exists():
    os.makedirs(output_folder)

# Open the ROS bag
with rosbags.Bag(bag_path) as bag:
    # Iterate through messages in the specified topic
    for topic, msg, t in bag.read_messages(topics=['/ipx_ros/image_raw']):
        # Assuming msg is a sensor_msgs/Image, convert to an OpenCV image
        # Note: You might need to adjust the conversion based on the actual message type
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

        # Construct filename and save the image
        filename = output_folder / f'frame_{t}.png'
        cv2.imwrite(str(filename), frame)
