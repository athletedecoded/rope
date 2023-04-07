# ROPE: Real-time Operating-Room Pose-Estimation on the Edge

Adapted from [MoveNet on Pi example](https://github.com/tensorflow/examples/tree/6d5dfdca227b64ea68c6a58f532666e5822764a0/lite/examples/pose_estimation/raspberry_pi) and [Pose Classification Tutorial](https://www.tensorflow.org/lite/tutorials/pose_classification)

*   Pose estimation: Detect keypoints, such as eye, ear, arm etc., from an input
    image.
    *   Input: An image
    *   Output: A list of keypoint coordinates and confidence score.

## Original MVOR Dataset & Annotations

```
# Images
wget https://s3.unistra.fr/camma_public/datasets/mvor/camma_mvor_dataset.zip
unzip -q camma_mvor_dataset.zip && rm camma_mvor_dataset.zip

# Annotations
cd camma_mvor_dataset
wget https://raw.githubusercontent.com/CAMMA-public/MVOR/master/annotations/camma_mvor_2018.json
```

## Modified ROPE MVOR Dataset

1. Download and unzip modified MVOR datatset into root i.e.
```
ROPE
|- eval
|- ml
   |...
|- mvor
    |- day1
    |...
    |- day4
    |- annotations.json
...
```

2. Generate ROPE formatted annotations `rope_gt.json`

```
python convert_gt.py --annot_path ./camma_mvor_dataset/camma_mvor_2018.json --out_dir mvor
```

## Install

```
python3 -m venv ~/.venv
source ~/.venv/bin/activate
sh setup.sh
```

## Inference

```
python3 inference.py --tracker <TRACKER> --threshold <DETECTION_THRESHOLD>
```

* `<TRACKER>` is pose tracker to use. Options: `bounding_box` (default) or `keypoint`
* `<DETECTION_THRESHOLD>` is threshold value (float) for all keypts to qualify as detected pose (default = 0.1: 0 < threshold < 1.0

## Evaluation


**Test original MVOR OpenPose results**

```
# MVOR x OpenPose bounding box detections
wget https://raw.githubusercontent.com/CAMMA-public/MVOR/799ec8c709624c6bbc8b6c88accb2192e15a88a6/detections_results/openpose_bbox.json
# MVOR x OpenPose keypoint detections
wget https://raw.githubusercontent.com/CAMMA-public/MVOR/799ec8c709624c6bbc8b6c88accb2192e15a88a6/detections_results/openpose_kps.json
# Run AP bounding box evaluation. Requires pycocotools. Run "pip install pycocotools" if not installed
python3 eval/ap.py --gt camma_mvor_dataset/camma_mvor_2018.json --dt openpose_bbox.json
# Run PCK evaluation
python3 eval/pck.py --gt camma_mvor_dataset/camma_mvor_2018.json --dt openpose_kps.json
```

**To evaluate MoveNet predictions**

```
# Run AP bounding box evaluation. Requires pycocotools. Run "pip install pycocotools" if not installed
python3 eval/ap.py --gt camma_mvor_dataset/camma_mvor_2018.json --dt eval_preds.json
# Run PCK evaluation
python3 eval/pck.py --gt camma_mvor_dataset/camma_mvor_2018.json --dt eval_preds.json
```

## Visualization

**Visualize IRT MoveNet Predictions (COCO Format)**

```
# Set IMG_DIR line 21
python3 viz_movenet.py --tracker <TRACKER>
```

* `<TRACKER>` is pose tracker to use. Options: `bounding_box` (default) or `keypoint`

**Visualize MVOR ground truth annotations (CAMMA Format)**

```
# Requires original camma_mvor_dataset and annotations

python3 viz_mvor.py \
       --inp_json camma_mvor_dataset/camma_mvor_2018.json \
       --img_dir camma_mvor_dataset \
       --show_ann true \
       --viz_2D true
```


**Visualize ROPE predictions vs MVOR ground truth (CAMMA Format)**

```
python viz_preds.py --annots <path_to_rope_gt> --preds <path_to_rope_preds> --day <dam_num> --cam <cam_num>
# ie. python viz_preds.py --annots mvor/rope_gt.json --preds viz_preds.json --day 1 --cam 1
```