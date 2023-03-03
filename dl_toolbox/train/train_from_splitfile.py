from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
#from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from dl_toolbox.lightning_modules import *
from dl_toolbox.lightning_datamodules import *
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
from dl_toolbox.torch_datasets import *
import dl_toolbox.callbacks as callbacks


"""
Changer data_path, labels, splitfile_path, epoch_len, save_dir... en fonction de l'expé.
"""

data_root = Path('/mnt/d/pfournie/Documents/data')
home = Path('/home/pfournie')

### Building training and validation datasets to concatenate from the splitfile lines
val_sets, train_sets = [], []
data_path = data_root / 'DIGITANIE'
split_name = 'toulouse'
splitfile_path = home / f'dl_toolbox/dl_toolbox/lightning_datamodules/splits/digitanie/{split_name}.csv'
dataset_factory = DatasetFactory()
val_folds=[8,9]
train_folds=[0,1]
with open(splitfile_path, newline='') as splitfile:
    reader = csv.reader(splitfile)
    next(reader)
    for row in reader:
        ds_name, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
        if int(fold) in val_folds+train_folds:
            is_val = int(fold) in val_folds
            window = Window(
                col_off=int(x0),
                row_off=int(y0),
                width=int(w),
                height=int(h)
            )
            ds = dataset_factory.create(ds_name)(
                image_path=data_path/image_path,
                label_path=data_path/label_path if label_path else None,
                tile=window,
                crop_size=256,
                fixed_crops=True if is_val else False,
                img_aug=None if is_val else 'd4',
                labels='6mainFuseVege',
                mins=ast.literal_eval(mins),
                maxs=ast.literal_eval(maxs)
            )
            if is_val:
                val_sets.append(ds)
            else:
                train_sets.append(ds)

### Building dataloaders
train_set = ConcatDataset(train_sets)
train_dataloaders = {}
train_dataloaders['sup'] = DataLoader(
    dataset=train_set,
    batch_size=8,
    collate_fn=CustomCollate(),
    sampler=RandomSampler(
        data_source=train_set,
        replacement=True,
        num_samples=5000
    ),
    num_workers=1,
    pin_memory=True,
    drop_last=True
)
val_set = ConcatDataset(val_sets)
val_dataloader = DataLoader(
    dataset=val_set,
    shuffle=False,
    collate_fn=CustomCollate(),
    batch_size=8,
    num_workers=1,
    pin_memory=True
)

### Building lightning module
module = CE(
    ignore_zero=True,
    #mixup=0.4, # incompatible with ignore_zero=True
    #network='Vgg',
    network='SmpUnet',
    encoder='efficientnet-b0',
    pretrained=False,
    weights=[],
    in_channels=4,
    out_channels=6,
    initial_lr=0.001,
    ttas=['vflip']
    #alphas=(0., 1.),
    #ramp=(0, 40000),
    #pseudo_threshold=0.9,
    #consist_aug='color-3',
    #emas=(0.9, 0.999)
)

### Metrics and plots from confmat callback
metrics_from_confmat = callbacks.MetricsFromConfmat(        
    num_classes=6,
    class_names=list(val_set.datasets[0].labels.keys())
)

### Trainer instance
trainer = Trainer(
    max_steps=50000,
    accelerator='cpu',
    devices=1,
    multiple_trainloader_mode='min_size',
    num_sanity_val_steps=0,
    limit_train_batches=2,
    limit_val_batches=2,
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
