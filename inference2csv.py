import argparse
import logging
import sys
import os
import time
import csv
import cv2
import glob
import time

from ml import MoveNetMultiPose

def run(tracker_type: str, detection_threshold: float) -> None:
    """Run inference on images in IMG_DIR.
    Args:
        tracker_type: Type of Tracker('keypoint' or 'bounding_box').
        detection_threshold: Only keep images with all landmark confidence score above this threshold.
    """

    # Define Variables
    num_days = 4
    num_cams = 3
    latency = 0
    total_frames = 0

    # Preprocess image paths
    MVOR_DIR = os.path.join(os.getcwd(),'mvor')

    # Initialize the pose estimator selected.
    pose_detector = MoveNetMultiPose('movenet_multipose', tracker_type)

    # Detect landmarks in each image and write it to a CSV file
    with open("predictions.csv", 'w') as csv_out_file:
        writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        headers = [
            'day_id',
            'cam_id',
            'img_id',
            'person_id',
            'x1',
            'y1',
            'x2',
            'y2',
            'nose_x', 
            'nose_y', 
            'nose_score', 
            'left_eye_x', 
            'left_eye_y', 
            'left_eye_score', 
            'right_eye_x', 
            'right_eye_y', 
            'right_eye_score', 
            'left_ear_x', 
            'left_ear_y', 
            'left_ear_score', 
            'right_ear_x',
            'right_ear_y',
            'right_ear_score',
            'left_shoulder_x',
            'left_shoulder_y',
            'left_shoulder_score',
            'right_shoulder_x',
            'right_shoulder_y',
            'right_shoulder_score',
            'left_elbow_x',
            'left_elbow_y',
            'left_elbow_score',
            'right_elbow_x',
            'right_elbow_y',
            'right_elbow_score',
            'left_wrist_x',
            'left_wrist_y',
            'left_wrist_score',
            'right_wrist_x',
            'right_wrist_y',
            'right_wrist_score',
            'left_hip_x',
            'left_hip_y',
            'left_hip_score',
            'right_hip_x',
            'right_hip_y',
            'right_hip_score',
            'left_knee_x',
            'left_knee_y',
            'left_knee_score',
            'left_knee_x',
            'left_knee_y',
            'left_knee_score',
            'right_ankle_x',
            'right_ankle_y',
            'right_ankle_score',
            'right_ankle_x'
            'right_ankle_y'
            'right_ankle_score'
        ]
        writer.writerow(headers)

        for day_num in range(1, num_days + 1):
            for cam_num in range(1, num_cams + 1):
                dir_path = os.path.join(MVOR_DIR,f'day{day_num}',f'cam{cam_num}','*png')
                frames = glob.glob(dir_path)
                for frame in frames:
                    img_name = frame.split('/')[-1]
                    img_id, ext = img_name.split('.')
                    image = cv2.imread(frame)

                    # Record initial time and number of iterations
                    init_time = time.time()
                    total_frames += 1

                    # Flip across y axis (?)
                    image = cv2.flip(image, 1)
                    # Run pose estimation using a MultiPose model
                    detect_persons = pose_detector.detect(image)
                    # Write keypts for each person to csv if all their landmarks were detected
                    for pid, person in enumerate(detect_persons):
                        min_score = min([keypoint.score for keypoint in person.keypoints])
                        if min_score > float(detection_threshold):
                            keypts = []
                            for bodypart in person.keypoints:
                                keypts.extend([bodypart.coordinate.x, bodypart.coordinate.y, bodypart.score])
                            # Write the landmark and bbox coordinates to its per-class CSV file
                            bbox = [
                                person.bounding_box.start_point.x, 
                                person.bounding_box.start_point.y, 
                                person.bounding_box.end_point.x,
                                person.bounding_box.end_point.y
                            ]
                            writer.writerow([day_num, cam_num, img_id, pid] + bbox + keypts)
                    
                    latency += time.time() - init_time
        latency /= total_frames # Normalize by number of frames
    print(f"Average Latency: {latency:.2f}")

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--tracker',
            help='Type of tracker to track poses across frames.',
            required=False,
            default='bounding_box')
    parser.add_argument(
            '--threshold',
            help='Detection threshold for keypoints.',
            required=False,
            default='0.1')
    
    args = parser.parse_args()

    run(args.tracker,args.threshold)

if __name__ == '__main__':
    main()