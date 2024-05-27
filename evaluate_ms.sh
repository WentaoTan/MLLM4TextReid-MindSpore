#!/bin/bash
export CUDA_HOME=/export/home/tanwentao1/local/cuda-11.7
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python evaluate_ms.py \
--name testing \
--img_aug \
--batch_size 16 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id+mlm' \
--num_epoch 60 \
--root_dir /export/home/tanwentao1/data
