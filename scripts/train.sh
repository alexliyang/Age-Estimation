#!/usr/bin/env bash

cd ../src/age_estimation/

python train.py \
--dataset_name 'WIKI_dataset' \
--log1p_target True \
--seed 17 \
--model_name 'se_resnext50_32_4' \
--se_type 'scSE' \
--se_integration 'standard' \
--reduction 16 \
--epochs 10 \
--batch_size 32 \
--optimizer_type 'adam' \
--decay 0.00006 \
--learning_rate 0.1 \
--augmentations False \
--early_stopping_patience 3 \
--alias ''