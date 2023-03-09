from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path, PurePath
from datetime import datetime
import os
from argparse import ArgumentParser
from pathlib import Path
import csv
import ast 
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils


if os.uname().nodename == 'WDTIS890Z': 
    data_root = Path('/mnt/d/pfournie/Documents/data')
    home = Path('/home/pfournie')
elif os.uname().nodename == 'qdtis056z': 
    data_root = Path('/data')
    home = Path('/d/pfournie')
    
data_path = data_root / 'SemCity-Toulouse-bench'
split_name = 'semcity_16tiles'
splitfile_path = home / f'dl_toolbox/dl_toolbox/datamodules/{split_name}.csv'
crop_size=256
img_aug = 'd4_color-3'
labels = 'base'
bands = [4,3,2]

batch_size = 4
num_samples = 2000
num_workers=4

mixup=0. # incompatible with ignore_zero=True
weights = [1,1,1,1,1,1,1,1]
network='SmpUnet'
encoder='efficientnet-b0'
pretrained=False
in_channels=len(bands)
out_channels=8
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(2,4,0.,2.)
pseudo_threshold=0.9
consist_aug='d4'
ema_ramp=utils.SigmoidRamp(3,6,0.9,0.99)

num_classes = 8



for train_folds, val_folds in [([0,1,2,3,4],[5,6])]:
    for mixup in [0]:
        for weights in [[1,1,1,1,1,1,1,1]]:
            
            datamodule = FromSplitfile()

            

            ### dataloaders
            
            train_set = ConcatDataset(train_sets)
            train_dataloaders = {}
            train_dataloaders['sup'] = DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                collate_fn=collate.CustomCollate(),
                sampler=RandomSampler(
                    data_source=train_set,
                    replacement=True,
                    num_samples=num_samples
                ),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            unsup_data = True
            if unsup_data:
                unsup_train_set = ConcatDataset(unsup_train_sets)
                train_dataloaders['unsup'] = DataLoader(
                    dataset=unsup_train_set,
                    batch_size=batch_size,
                    collate_fn=collate.CustomCollate(),
                    sampler=RandomSampler(
                        data_source=unsup_train_set,
                        replacement=True,
                        num_samples=num_samples
                    ),
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True
                )
                
            val_set = ConcatDataset(val_sets)
            val_dataloader = DataLoader(
                dataset=val_set,
                shuffle=False,
                collate_fn=collate.CustomCollate(),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True
            )

            ### Building lightning module
            module = modules.MeanTeacher(
                mixup=mixup, # incompatible with ignore_zero=True
                network=network,
                encoder=encoder,
                pretrained=pretrained,
                weights=weights,
                in_channels=in_channels,
                out_channels=out_channels,
                initial_lr=initial_lr,
                ttas=ttas,
                alpha_ramp=alpha_ramp,
                pseudo_threshold=pseudo_threshold,
                consist_aug=consist_aug,
                ema_ramp=ema_ramp
            )

            ### Metrics and plots from confmat callback
            metrics_from_confmat = callbacks.MetricsFromConfmat(        
                num_classes=num_classes,
                class_names=class_names
            )

            ### Trainer instance
            trainer = Trainer(
                max_steps=30000,
                accelerator='gpu',
                devices=1,
                multiple_trainloader_mode='min_size',
                num_sanity_val_steps=0,
                limit_train_batches=1,
                limit_val_batches=1,
                logger=TensorBoardLogger(
                    save_dir=data_root / 'outputs/DIGITANIE',
                    name=split_name,
                    version=f'{datetime.now():%d%b%y-%Hh%Mm%S}'
                ),
                callbacks=[
                    ModelCheckpoint(),
                    metrics_from_confmat
                ]
            )
                
            #ckpt_path='/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'
            trainer.fit(
                model=module,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloader,
                ckpt_path=None
            )
