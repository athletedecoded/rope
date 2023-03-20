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
sh setup.sh
```
## Run the pose estimation sample with visualisation

```
# Set IMG_DIR line 21
python3 visualizer.py --tracker <TRACKER>
```

* `<TRACKER>` is pose tracker to use. Options: `bounding_box` (default) or `keypoint`

## Run the pose estimation to generate annotation file

```
python3 inference.py --tracker <TRACKER> --threshold <DETECTION_THRESHOLD>
```

* `<TRACKER>` is pose tracker to use. Options: `bounding_box` (default) or `keypoint`
* `<DETECTION_THRESHOLD>` is threshold value (float) for all keypts to qualify as detected pose: 0 < threshold < 1.0