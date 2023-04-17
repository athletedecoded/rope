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
from dataloader import DataLoader

from ml import MoveNetMultiPose
from utils import coco_to_camma_kps

def run(tracker_type: str, detection_threshold: float, fps: str) -> None:
    """Run inference on images in IMG_DIR.
    Args:
        tracker_type: Type of Tracker('keypoint' or 'bounding_box').
        detection_threshold: Only keep images with all landmark confidence score above this threshold.
    """
    print(f'Running ROPE inference (COCO UB) -- tracker: {tracker_type}, detection threshold: {detection_threshold}') 

    # Define Variables
    num_days = 4
    num_cams = 3
    latency = 0
    total_frames = 0

    fps = int(fps)

    # Preprocess image paths
    MVOR_DIR = os.path.join(os.getcwd(),'mvor')

    # Initialize the pose estimator selected.
    pose_detector = MoveNetMultiPose('movenet_multipose', tracker_type)

    # Write predictions and bbox files in json format -- eval and viz formats
    preds_eval = []
    preds_viz = {}

    # Initialize dataloader
    dl = DataLoader()
    dl.load_images(num_days, num_cams)

    # Counter for the frame number
    image_index = 0
    bbox_counter = 0
    annot_counter = 0

    for day_num in range(1, num_days + 1):
        for cam_num in range(1, num_cams + 1):
            dir_path = os.path.join(MVOR_DIR,f'day{day_num}',f'cam{cam_num}','*png')
            frames = glob.glob(dir_path)
            for frame in frames:
                # Record initial time and number of iterations
                init_time = time.time()
                total_frames += 1
                img_name = frame.split('/')[-1]
                img_num, ext = img_name.split('.')
                img_id = f'{day_num}00{cam_num}0{img_num}'
                # Load frame as image
                image = dl.get_item(image_index)
                image_index += 1
                # image = cv2.flip(image, 1)
                # Run pose estimation using a MultiPose model
                detect_persons = pose_detector.detect(image)
                # Init default vals for undetected person
                bbox_only = 1
                camma_kpts = [0]*30 # 10 pts x 3 vals
                coco_kpts = [0]*51 # 17 pts x 3 vals
                # Write predictions for each person
                for pid, person in enumerate(detect_persons):
                    min_score = min([keypoint.score for keypoint in person.keypoints[:13]])
                    if min_score >= float(detection_threshold):
                        annot_counter += 1
                        bbox_only = 0
                        # Extract coco_kpts and convert to camma
                        raw_kpts = []
                        for bodypart in person.keypoints:
                            raw_kpts.append([bodypart.coordinate.x, bodypart.coordinate.y, bodypart.score])
                        np_kpts = np.array(raw_kpts)
                        camma_kpts = [float(_a) for _a in coco_to_camma_kps(np_kpts).reshape(-1).flatten().tolist()]
                        coco_kpts = [float(_a) for _a in np_kpts.reshape(-1).flatten().tolist()]
                    # Extract person score
                    person_score = person.score
                    # Extract bounding box w format: [x1, x2, w, h]
                    bbox = [
                        person.bounding_box.start_point.x, 
                        person.bounding_box.start_point.y, 
                        person.bounding_box.end_point.x - person.bounding_box.start_point.x,
                        person.bounding_box.end_point.y - person.bounding_box.start_point.y
                    ]
                    bbox_counter += 1
                    # update pid if bbox only
                    if bbox_only:
                        pid = -1
                    # Write to preds_viz format for visualization
                    # Check if img_id key exists in preds_viz
                    if img_id in preds_viz:
                        # append new person data
                        preds_viz[img_id].append({
                        "person_id": int(pid),
                        "category_id": "1",
                        "bbox": bbox, 
                        "bbox_only": bbox_only,
                        "keypoints":camma_kpts,
                        "coco_keypoints":coco_kpts,
                        "score": float(person_score)
                    })
                    else:
                        preds_viz[img_id] = [{
                        "person_id": int(pid),
                        "category_id": "1",
                        "bbox": bbox, 
                        "bbox_only": bbox_only,
                        "keypoints":camma_kpts,
                        "coco_keypoints":coco_kpts,
                        "score": float(person_score)
                    }]
                    # Write to preds_eval format for evaluation
                    preds_eval.append({
                        "image_id": int(img_id),
                        "person_id": int(pid),
                        "category_id": "1",
                        "bbox": bbox, 
                        "bbox_only": bbox_only,
                        "keypoints":camma_kpts,
                        "coco_keypoints":coco_kpts,
                        "score": float(person_score)
                    })
                latency += time.time() - init_time
                # Buffer a frame for some time based on the FPS
                while time.time() - init_time < 1/fps:
                    pass
    latency /= total_frames # Normalize by number of frames
    print(f"Average Latency: {latency:.2f}")
    print("Input FPS --> Actual FPS")
    print(f"\t{fps} --> {1/latency:.3f}")
    # Write preds
    print(f"Writing to ./preds_viz_cocoUB_{tracker_type}_{detection_threshold}.json\n")    
    with open(f"preds_viz_cocoUB_{tracker_type}_{detection_threshold}.json", "w") as f:
        json.dump(preds_viz, f)
    print(f"Writing to ./preds_eval_cocoUB_{tracker_type}_{detection_threshold}.json\n")    
    with open(f"preds_eval_cocoUB_{tracker_type}_{detection_threshold}.json", "w") as f:
        json.dump(preds_eval, f)

    print(f"Number of bbox detections = {bbox_counter}")
    print(f"Number of person detections = {annot_counter}")

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--tracker',
            help='Type of tracker to track poses across frames.',
            required=False,
            default='bbox')
    parser.add_argument(
            '--threshold',
            help='Detection threshold for keypoints.',
            required=False,
            default='0.1')
    parser.add_argument(
            '--fps',
            help='Ideal FPS (will be ignored if less than max fps).',
            required=False,
            default='9')
    
    args = parser.parse_args()

    run(args.tracker, args.threshold, args.fps)

if __name__ == '__main__':
    main()