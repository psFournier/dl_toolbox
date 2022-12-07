#!/bin/bash

"${HOME}"/dl_toolbox/venv/bin/python \
"${HOME}"/dl_toolbox/dl_toolbox/train/train.py \
--data_path /d/pfournie/ai4geo/data/miniworld_tif \
--splitfile_path "${HOME}"/ai4geo/splits/mw/christchurch.csv \
--datamodule Splitfile \
--labels base \
--crop_size 256 \
--img_aug d4 \
--test_folds 0 \
--train_folds 5 6 7 8 \
--batch_size 4 \
--workers 6 \
--max_epochs 10 \
--epoch_len 2000 \
--model CE_MT \
--final_alpha 1 \
--alpha_milestones 0 3 \
--pseudo_threshold 0.9 \
--unsup_train_folds $(seq 0 9) \
--unsup_img_aug d4 \
--consist_aug cutmix2-3 \
--ema 0.99 \
--weights 1 2 \
--network SmpUnet \
--encoder efficientnet-b0 \
--in_channels 3 \
--out_channels 2 \
--initial_lr 0.001 \
--final_lr 0.0005 \
--lr_milestones 30 \
--exp_name test_ce_christchurch \
--output_dir "${HOME}"/ai4geo/outputs \
--multiple_trainloader_mode min_size \
--limit_train_batches 1 \
--limit_val_batches 1 \
--gpu 1

#--ignore_index -1 \

#--data_path "${TMPDIR}"/digitanie \

#--final_alpha 1 \
#--alpha_milestones 1 3 \
#--pseudo_threshold 0.9 \
#--unsup_train_folds $(seq 0 9) \
#--final_alpha 1 \
#--alpha_milestones 50 100 \
#--pseudo_threshold 0.9 \
#--unsup_splitfile_path "${SPLITS}"/split_digitanie_toulouse_unlabeled.csv \
#--unsup_batch_size 32 \
#--unsup_train_folds $(seq 0 9) \

#--ema 0.99 \

#--limit_train_batches 1 \
#--limit_val_batches 1 \
