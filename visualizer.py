import argparse
import logging
import sys
import os
import time

import cv2
import glob
from ml import Movenet
from ml import MoveNetMultiPose
import utils


def run(tracker_type: str) -> None:
  """Run inference on images in IMG_DIR.

  Args:
    tracker_type: Type of Tracker('keypoint' or 'bounding_box').
  """

  IMG_DIR = "mvor/day1/cam1/"

  # Initialize the pose estimator selected.
  pose_detector = MoveNetMultiPose('movenet_multipose', tracker_type)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  classification_results_to_show = 3
  fps_avg_frame_count = 10
  keypoint_detection_threshold_for_classifier = 0.1
  classifier = None

  # Stream images
  data_path = os.path.join(os.getcwd(),IMG_DIR,'*png') 
  files = glob.glob(data_path)

  # Set file display
  cv2.namedWindow("ROPE", cv2.WINDOW_NORMAL)

  for frame in files:
    image = cv2.imread(frame)
    counter += 1
    # Flip across y axis (?)
    image = cv2.flip(image, 1)

    # Run pose estimation using a MultiPose model.
    list_persons = pose_detector.detect(image)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, list_persons)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow("ROPE", image)

  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--tracker',
      help='Type of tracker to track poses across frames.',
      required=False,
      default='bounding_box')
  
  args = parser.parse_args()

  run(args.tracker)

if __name__ == '__main__':
  main()
