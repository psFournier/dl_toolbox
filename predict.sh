#!/bin/bash

PYTHON=/d/pfournie/ai4geo/venv/bin/python
#SCRIPT=/d/pfournie/dl_toolbox/dl_toolbox/inference/infer_test_set_from_split.py
SCRIPT=/d/pfournie/dl_toolbox/dl_toolbox/inference/infer_one_image.py
LOGS=/d/pfournie/ai4geo/outputs/digitanie/version_59

"${PYTHON}" "${SCRIPT}" \
--ckpt_path "${LOGS}"/checkpoints/epoch=141-step=22293.ckpt \
--dataset semcity \
--workers 4 \
--batch_size 8 \
--num_classes 11 \
--in_channels 3 \
--crop_size 256 \
--crop_step 256 \
--encoder efficientnet-b5 \
--train_with_void \
--image_path /d/pfournie/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif \
--tile 0 0 6000 6000 \
--label_path /d/pfournie/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif


#--splitfile_path /d/pfournie/ai4geo/split_scenario_1b.csv \
#--data_path /d/pfournie/ai4geo/data/DIGITANIE \
#--write_probas 
#--output_path "${LOGS}"/probas_nopostproc \
#--test_fold 1 \


#--tta hflip vflip rot90 d1flip d2flip rot180 rot270 color color \
#--image_path /d/pfournie/ai4geo/data/DIGITANIE/Strasbourg/strasbourg_tuile_4_img_normalized.tif \
#--output_probas "${LOGS}"/strasbourg_tuile_4_probas.tif \
#--output_preds "${LOGS}"/strasbourg_tuile_4_preds.tif \
#--output_errors "${LOGS}"/strasbourg_tuile_4_errors.tif \
#--label_path /d/pfournie/ai4geo/data/DIGITANIE/Strasbourg/strasbourg_tuile_4.tif \
#--tile 0 1000 1000 1000



