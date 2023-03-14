from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path, PurePath
from datetime import datetime
import os
import csv
from argparse import ArgumentParser
from pathlib import Path
import ast 
from pytorch_lightning import LightningDataModule
import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils



"""
Changer data_path, labels, splitfile_path, epoch_len, save_dir... en fonction de l'expé.
"""


if os.uname().nodename == 'WDTIS890Z': 
    data_root = Path('/mnt/d/pfournie/Documents/data')
    home = Path('/home/pfournie')
    save_root = data_root / 'outputs'
elif os.uname().nodename == 'qdtis056z': 
    data_root = Path('/data')
    home = Path('/d/pfournie')
    save_root = data_root / 'outputs'
else:
    data_root = Path('/work/OT/ai4geo/DATA/DATASETS')
    home = Path('/home/eh/fournip')
    save_root = Path('/work/OT/ai4usr/fournip') / 'outputs'
    print('torch device count', torch.cuda.device_count())

data_path = data_root / 'DIGITANIE'
split = 'toulouse'
split_dir = home / f'dl_toolbox/dl_toolbox/datamodules'
train_split = (split_dir/'digitanie_toulouse.csv', [0,1,2,3,4])
unsup_train_split = (split_dir/'digitanie_toulouse_big.csv', [0])
val_split = (split_dir/'digitanie_toulouse.csv', [5,6])
crop_size=256
crop_step=256
aug = 'd4_color-3'
unsup_aug = 'd4'
labels = 'mainFuseVege'
bands = [1,2,3]

batch_size = 4
num_samples = 2000
num_workers=4

mixup=0. # incompatible with ignore_zero=True
weights = [1,1,1,1,1,1,]
network='SmpUnet'
encoder='efficientnet-b0'
pretrained=False
in_channels=len(bands)
out_channels=6
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(2,4,0.,2.)
pseudo_threshold=0.9
consist_aug='d4'
ema_ramp=utils.SigmoidRamp(3,6,0.9,0.99)

save_dir = save_root / 'DIGITANIE'
log_name = 'toulouse'

class_names = [label.name for label in datasets.Digitanie.possible_labels[labels].value.labels]
class_num = datasets.Digitanie.possible_labels[labels].value.num


dataset_factory = datasets.DatasetFactory()

def from_csv(data_path, split_path, folds):
    
    with open(split_path, newline='') as splitfile:
        reader = csv.reader(splitfile)
        next(reader)
        for row in reader:
            name, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
            if int(fold) in folds:
                window = Window(
                    col_off=int(x0),
                    row_off=int(y0),
                    width=int(w),
                    height=int(h)
                )
                yield dataset_factory.create(name)(
                    image_path=data_path/image_path,
                    label_path=data_path/label_path if label_path != 'none' else None,
                    mins=np.array(ast.literal_eval(mins)).reshape(-1, 1, 1),
                    maxs=np.array(ast.literal_eval(maxs)).reshape(-1, 1, 1),
                    window=window
                )

                
train_sets = from_csv(data_path, *train_split)
train_set = ConcatDataset(
    [datasets.Raster(ds, crop_size, aug, bands, labels) for ds in train_sets]
)
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

if unsup_train_split:
    unsup_train_sets = from_csv(data_path, *unsup_train_split)
    unsup_train_set = ConcatDataset(
        [datasets.Raster(ds, crop_size, unsup_aug, bands, labels) for ds in unsup_train_sets]
    )
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

val_sets = from_csv(data_path, *val_split)
val_set = ConcatDataset(
    [datasets.PretiledRaster(ds, crop_size, crop_step, aug=None, bands=bands, labels=labels) for ds in val_sets]
)
val_dataloader = DataLoader(
    dataset=val_set,
    shuffle=False,
    collate_fn=collate.CustomCollate(),
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True
)

            
#with open(splitfile_path, newline='') as splitfile:
#    reader = csv.reader(splitfile)
#    next(reader)
#    for name, _, img, label, x0, y0, w, h, fold, mins, maxs in reader:
#
#        if int(fold) in folds:
#            window = Window(int(x0), int(y0), int(w), int(h))
#            args['tile'] = window
#            args['image_path'] = data_path/image_path
#            args['label_path'] = data_path/label_path if label_path else None
#            args['mins'] = ast.literal_eval(mins)
#            args['maxs'] = ast.literal_eval(maxs)
#            yield class_name, args.copy()
#
## datasets
#
#train_sets = []
#for ds_name, args in gen_dataset_args_from_splitfile(
#    splitfile_path,
#    data_path,
#    train_folds
#):
#    ds = dataset_factory.create(ds_name)(
#        crop_size=crop_size,
#        fixed_crops=False,
#        img_aug=img_aug,
#        labels=labels,
#        bands=bands,
#        **args
#    )
#    train_sets.append(ds)
#
#class_names = list(train_sets[0].labels.keys())
#
#val_sets = []
#for ds_name, args in gen_dataset_args_from_splitfile(
#    splitfile_path,
#    data_path,
#    val_folds
#):
#    ds = dataset_factory.create(ds_name)(
#        crop_size=crop_size,
#        fixed_crops=True,
#        img_aug=None,
#        labels=labels,
#        bands=bands,
#        **args
#    )
#    val_sets.append(ds)
#
#### dataloaders
#
#train_set = ConcatDataset(train_sets)
#train_dataloaders = {}
#train_dataloaders['sup'] = DataLoader(
#    dataset=train_set,
#    batch_size=batch_size,
#    collate_fn=collate.CustomCollate(),
#    sampler=RandomSampler(
#        data_source=train_set,
#        replacement=True,
#        num_samples=num_samples
#    ),
#    num_workers=num_workers,
#    pin_memory=True,
#    drop_last=True
#)
#
#unsup_data = True            
#if unsup_data:
#    unsup_train_sets = []
#    for ds_name, args in gen_dataset_args_from_splitfile(
#        splitfile_path,
#        data_path,
#        list(range(10))
#    ):
#        ds = dataset_factory.create(ds_name)(
#            crop_size=crop_size,
#            fixed_crops=False,
#            img_aug=img_aug,
#            labels=labels,
#            bands=bands,
#            **args
#        )
#        unsup_train_sets.append(ds)
#    unsup_train_set = ConcatDataset(unsup_train_sets)
#    train_dataloaders['unsup'] = DataLoader(
#        dataset=unsup_train_set,
#        batch_size=batch_size,
#        collate_fn=collate.CustomCollate(),
#        sampler=RandomSampler(
#            data_source=unsup_train_set,
#            replacement=True,
#            num_samples=num_samples
#        ),
#        num_workers=num_workers,
#        pin_memory=True,
#        drop_last=True
#    )
#
#val_set = ConcatDataset(val_sets)
#val_dataloader = DataLoader(
#    dataset=val_set,
#    shuffle=False,
#    collate_fn=collate.CustomCollate(),
#    batch_size=batch_size,
#    num_workers=num_workers,
#    pin_memory=True
#)

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
    num_classes=class_num,
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
        save_dir=save_dir,
        name=log_name,
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
