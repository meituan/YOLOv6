#!/bin/bash
# This is a sample bash script to run YOLOv6 training.

python tools/train.py \
    --batch 64 \
    --conf configs/yolov6s_finetune.py \
    --data data/human.yaml \
    --img-size 832 \
    --epochs 200 \
    --workers 4 \
    --eval-interval 5 \
    --heavy-eval-range 10 \
    --output-dir ./runs/train/832_jpg_date \
    --write_trainbatch_tb True
