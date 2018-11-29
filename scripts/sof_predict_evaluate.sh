#!/usr/bin/env bash

cd ../src/age_estimation/

python sof_predict_evaluate.py \
--log1p_target True \
--weights '../../nn_models/best_SE-ResNeXt-50 (32 x 4d).h5' \
--model_name 'se_resnext50_32_4' \
--se_type 'scSE' \
--se_integration 'standard' \
--reduction 16