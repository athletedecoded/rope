# ROPE: Real-time Operating-Room Pose-Estimation on the Edge

Adapted from [MoveNet on Pi example](https://github.com/tensorflow/examples/tree/6d5dfdca227b64ea68c6a58f532666e5822764a0/lite/examples/pose_estimation/raspberry_pi) and [Pose Classification Tutorial](https://www.tensorflow.org/lite/tutorials/pose_classification)

*   Pose estimation: Detect keypoints, such as eye, ear, arm etc., from an input
    image.
    *   Input: An image
    *   Output: A list of keypoint coordinates and confidence score.

## MVOR Dataset

Download and unzip MVOR datatset into root i.e.
```
ROPE
|- ml
   |...
|- mvor
    |- day1
    |...
    |- day4
    |- annotations.json
...

```

## Install

```
python3 -m venv ~/.venv
source ~/.venv/bin/activate
sh setup.sh
```
## Run the pose estimation sample with visualisation

```
# Set IMG_DIR line 21
python3 visualizer.py --tracker <TRACKER>
```

* `<TRACKER>` is pose tracker to use. Options: `bounding_box` (default) or `keypoint`

## Inference

```
python3 inference.py --tracker <TRACKER> --threshold <DETECTION_THRESHOLD>
```

* `<TRACKER>` is pose tracker to use. Options: `bounding_box` (default) or `keypoint`
* `<DETECTION_THRESHOLD>` is threshold value (float) for all keypts to qualify as detected pose (default = 0.0): 0 < threshold < 1.0

## Evaluation

**Test original MVOR OpenPose results**

```
# MVOR x OpenPose bounding box detections
wget https://raw.githubusercontent.com/CAMMA-public/MVOR/799ec8c709624c6bbc8b6c88accb2192e15a88a6/detections_results/openpose_bbox.json
# MVOR x OpenPose keypoint detections
wget https://raw.githubusercontent.com/CAMMA-public/MVOR/799ec8c709624c6bbc8b6c88accb2192e15a88a6/detections_results/openpose_kps.json
# Run AP bounding box evaluation. Requires pycocotools. Run "pip install pycocotools" if not installed
python3 eval/ap.py --gt mvor/annotations.json --dt openpose_bbox.json
# Run PCK evaluation
python3 eval/pck.py --gt mvor/annotations.json --dt openpose_kps.json
```

**To evaluate MoveNet predictions**

```
# Run AP bounding box evaluation. Requires pycocotools. Run "pip install pycocotools" if not installed
python3 eval/ap.py --gt mvor/annotations.json --dt predictions.json
# Run PCK evaluation
python3 eval/pck.py --gt mvor/annotations.json --dt predictions.json
```