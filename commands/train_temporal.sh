#!/usr/bin/env bash
set -euo pipefail

mkdir -p exps/logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 /home/yury/anaconda3/envs/TransVOD++/bin/python tools/train_rfdetr_temporal.py \
  --dataset-root data/car_dataset_temporal \
  --output-dir exps/rfdetr_full_temporal_surrounding3_step2_10ep \
  --variant small \
  --num-classes 1 \
  --resolution 640 \
  --temporal-num-ref-frames 3 \
  --temporal-fusion-layers 1 \
  --temporal-ref-frame-mode surrounding \
  --temporal-ref-frame-step 2 \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --lr 1e-4 \
  --lr-encoder 1e-5 \
  --lr-drop 8 \
  --num-workers 2 \
  2>&1 | tee exps/logs/rfdetr_full_temporal_surrounding3_step2_10ep.log
