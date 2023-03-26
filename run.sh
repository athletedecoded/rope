#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt

# Uncomment this if you also want to download the single person pose model
# FILE=${DATA_DIR}/movenet_lightning.tflite
# if [ ! -f "$FILE" ]; then
#   curl \
#     -L 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite' \
#     -o ${FILE}
# fi

FILE=${DATA_DIR}/movenet_multipose.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1?lite-format=tflite' \
    -o ${FILE}
fi

python3 inference.py

echo -e "MoveNet MultiPose model downloaded to ${DATA_DIR}"