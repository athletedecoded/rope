import argparse
import logging
import sys
import os
import json
import time
import csv
import cv2
import glob
import time
import numpy as np

from ml import MoveNetMultiPose
from utils import coco_to_camma_kps

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

    # Write predictions and bbox files in json format -- eval and viz formats
    eval_preds = []
    viz_preds = {}
    for day_num in range(1, num_days + 1):
        print(f'Day: {day_num}')
        for cam_num in range(1, num_cams + 1):
            print(f'Cam: {cam_num}')
            dir_path = os.path.join(MVOR_DIR,f'day{day_num}',f'cam{cam_num}','*png')
            frames = glob.glob(dir_path)
            for frame in frames:
                img_name = frame.split('/')[-1]
                img_num, ext = img_name.split('.')
                img_id = f'{day_num}00{cam_num}0{img_num}'
                # Record initial time and number of iterations
                init_time = time.time()
                total_frames += 1
                # Load frame as image
                image = cv2.imread(frame)
                # Flip across y axis (?)
                # image = cv2.flip(image, 1)
                # Run pose estimation using a MultiPose model
                detect_persons = pose_detector.detect(image)
                # Init default vals
                bbox_only = 1
                camma_kpts = None
                coco_kpts = None
                kpt_score = 0
                # Write predictions for each person
                for pid, person in enumerate(detect_persons):
                    min_score = min([keypoint.score for keypoint in person.keypoints])
                    if min_score > float(detection_threshold):
                        bbox_only = 0
                        # Extract coco_kpts and convert to camma
                        raw_kpts = []
                        for bodypart in person.keypoints:
                            raw_kpts.append([bodypart.coordinate.x, bodypart.coordinate.y, bodypart.score])
                        np_kpts = np.array(raw_kpts)
                        test = coco_to_camma_kps(np_kpts)
                        camma_kpts = [float(_a) for _a in coco_to_camma_kps(np_kpts).reshape(-1).flatten().tolist()]
                        coco_kpts = [float(_a) for _a in np_kpts.reshape(-1).flatten().tolist()]
                    # Extract person score
                    kpt_score = person.score
                    # Extract bounding box w format: [x1, x2, w, h]
                    bbox = [
                        person.bounding_box.start_point.x, 
                        person.bounding_box.start_point.y, 
                        person.bounding_box.end_point.x - person.bounding_box.start_point.x,
                        person.bounding_box.end_point.y - person.bounding_box.start_point.y
                    ]
                    # update pid if bbox only
                    if bbox_only:
                        pid = -1
                    # Write to viz_preds format
                    # Check if img_id key exists in viz_preds
                    if img_id in viz_preds:
                        # append new person data
                        viz_preds[img_id].append({
                        "person_id": int(pid),
                        "category_id": "1",
                        "bbox": bbox, 
                        "bbox_only": bbox_only,
                        "keypoints":camma_kpts,
                        "coco_keypoints":coco_kpts,
                        "score": float(kpt_score)
                    })
                    else:
                        viz_preds[img_id] = [{
                        "person_id": int(pid),
                        "category_id": "1",
                        "bbox": bbox, 
                        "bbox_only": bbox_only,
                        "keypoints":camma_kpts,
                        "coco_keypoints":coco_kpts,
                        "score": float(kpt_score)
                    }]
                    # Write to eval_preds format
                    eval_preds.append({
                        "image_id": int(img_id),
                        "person_id": int(pid),
                        "category_id": "1",
                        "bbox": bbox, 
                        "bbox_only": bbox_only,
                        "keypoints":camma_kpts,
                        "coco_keypoints":coco_kpts,
                        "score": float(kpt_score)
                    })
                latency += time.time() - init_time
    latency /= total_frames # Normalize by number of frames
    print(f"Average Latency: {latency:.2f}")
    print("Writing viz_preds to ./viz_preds.json\n")    
    with open("viz_preds.json", "w") as f:
        json.dump(viz_preds, f)
    print("Writing eval_preds to ./eval_preds.json\n")    
    with open("eval_preds.json", "w") as f:
        json.dump(eval_preds, f)

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
            default='0.0')
    
    args = parser.parse_args()

    run(args.tracker,args.threshold)

if __name__ == '__main__':
    main()