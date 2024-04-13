import cv2
import os

def frames_to_video(frames_dir, video_name, fps=24):
  """
  Converts a directory of image frames into an .mp4 video.

  Args:
      frames_dir (str): Path to the directory containing the image frames.
      video_name (str): Name of the output video file.
      fps (int, optional): Frames per second for the video. Defaults to 24.
  """

  # Get all image files in the directory
  images = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]

  # Ensure at least one image is present
  if not images:
    raise ValueError("No image files found in the directory!")

  # Read the first image to get frame dimensions
  first_frame = cv2.imread(images[0])
  frame_height, frame_width, _ = first_frame.shape

  # Create a video writer object
  video_writer = cv2.VideoWriter(f"{video_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

  # Write each frame to the video
  for image in images:
    frame = cv2.imread(image)
    video_writer.write(frame)

  # Release the video writer
  video_writer.release()

  print(f"Video created: {video_name}.mp4")

IMG_DIR = '/media/xdoestech/VT Storage 1/PERCEPTION_DATA_BAGS/test6_ucb_2023-11-21-16-08-14_imgs'
VIDEO_NAME = '12fps_ucb_2023_11_21_test5'
frames_to_video(IMG_DIR, VIDEO_NAME, fps=12)