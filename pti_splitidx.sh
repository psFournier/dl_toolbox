#!/bin/bash

DIR=/d/pfournie/ai4geo
DATA="${DIR}"/data
PYTHON="${DIR}"/venv/bin/python3
SCRIPT=/d/pfournie/dl_toolbox/dl_toolbox/train/train.py
SPLITS="${DIR}"/splits
OUTPUTS="${DIR}"/outputs

"${PYTHON}" "${SCRIPT}" \
--datamodule SplitIdxSemisup \
--model CE_PL \
--network Vgg \
--dataset Resisc \
--data_path "$DATA"/NWPU-RESISC45 \
--labels base \
--img_aug d4 \
--split 50 200 \
--n_unsup_img 700 \
--unsup_aug color-5 \
--sup_batch_size 16 \
--out_channels 45 \
--in_channels 3 \
--ignore_index -1 \
--alpha_milestones 40 70 \
--final_alpha 2 \
--pseudo_threshold 0.9 \
--max_epochs 200 \
--lr_milestones 150 \
--output_dir "${OUTPUTS}" \
--workers 6 \
--epoch_len 10000 \
--exp_name resisc \
--initial_lr 0.001 \
--final_lr 0.0005 \
--multiple_trainloader_mode min_size \
--gpu 1 



#--limit_train_batches 1 \
#--limit_val_batches 1

# --final_alpha 2 \
# --alpha_milestones 2 8 \
# --pseudo_threshold 0.7 \
# --unsup_splitfile_path "${SPLITS}"/split_semcity_finegrained.csv \
# --unsup_batch_size 8 \
# --unsup_train_folds 0 1 2 3 4 5 6 7 8 9 \

