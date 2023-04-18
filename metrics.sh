#!/bin/bash

## Parameter Tuning
# pip install pycocotools
# $TRACKER=bbox
# for THRESHOLD in 0.1 0.2 0.3; do
#     for MODEL in cocoUB camma50; do
#         # Run inference
#         python3 inference_$MODEL.py --tracker $TRACKER --threshold $THRESHOLD
#         # Run PCK evaluation
#         python3 eval/pck.py --gt camma_mvor_dataset/camma_mvor_2018.json --dt preds_eval_${MODEL}_${TRACKER}_${THRESHOLD}.json
#     done
# done

# # Final Evaluation
# pip install pycocotools
$THRESHOLD = 0.1
$TRACKER = bbox
for MODEL in cocoUB camma50; do
    # Run inference
    python3 inference_${MODEL}.py --tracker ${TRACKER} --threshold ${THRESHOLD}
    # Run PCK evaluation
    python3 eval/pck.py --gt camma_mvor_dataset/camma_mvor_2018.json --dt preds_eval_${MODEL}_${TRACKER}_${THRESHOLD}.json
done
# Run AP evaluation
python3 eval/ap.py --gt camma_mvor_dataset/camma_mvor_2018.json --dt preds_eval_${MODEL}_${TRACKER}_${THRESHOLD}.json


