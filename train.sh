#!/bin/bash

"${HOME}"/dl_toolbox/venv/bin/python \
"${HOME}"/dl_toolbox/dl_toolbox/train/train.py \
--max_epochs 50 \
--multiple_trainloader_mode min_size \
--gpu 1

#--limit_train_batches 1 \
#--limit_val_batches 1 \