#!/bin/bash

"${HOME}"/dl_toolbox/venv/bin/python \
"${HOME}"/dl_toolbox/dl_toolbox/train/train.py \
--datamodule SplitfileSup \
--data_path /data/digitanie_v2 \
--splitfile_path "${HOME}"/ai4geo/splits/Biarritz.csv \
--labels 6 \
--crop_size 256 \
--img_aug d4 \
--test_folds $(seq 0 3) \
--train_folds $(seq 4 9) \
--sup_batch_size 16 \
--workers 6 \
--max_epochs 50 \
--epoch_len 5000 \
--model CE \
--network SmpUnet \
--encoder efficientnet-b0 \
--in_channels 3 \
--out_channels 10 \
--ignore_index 0 \
--initial_lr 0.001 \
--final_lr 0.0005 \
--lr_milestones 30 \
--exp_name tou_unet_ce \
--output_dir "${HOME}"/ai4geo/outputs \
--multiple_trainloader_mode min_size \
--gpu 1

#--data_path "${TMPDIR}"/digitanie \


#--final_alpha 1 \
#--alpha_milestones 50 100 \
#--pseudo_threshold 0.9 \
#--unsup_splitfile_path "${SPLITS}"/split_digitanie_toulouse_unlabeled.csv \
#--unsup_batch_size 32 \
#--unsup_train_folds $(seq 0 9) \

#--ema 0.99 \

#--limit_train_batches 1 \
#--limit_val_batches 1 \
