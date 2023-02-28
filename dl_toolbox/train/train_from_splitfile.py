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

"""
Changer data_path, labels, splitfile_path, epoch_len, save_dir... en fonction de l'expé.
"""

### Datasets parameters
split='arcachon'
#data_path=Path(os.environ['TMPDIR'])/'DIGITANIE'
data_path=Path('/work/OT/ai4geo/DATA/DATASETS/DIGITANIE')
#data_path=Path('/scratchf/AI4GEO/DIGITANIE')
#data_path=Path('/data/DIGITANIE')
#data_path=Path('/home/pfournie/ai4geo/data/SemCity-Toulouse-bench')
labels='6mainFuseVege'
img_aug='d4'
crop_size=256
val_folds=[8,9]
train_folds=list(range(8))

### Dataloaders parameters
batch_size = 16
epoch_len=5000
workers=6

### Module parameters
out_channels=6
ignore_zero=False
#no_pred_zero=True
#mixup=0.4
#network='Vgg'
network='SmpUnet'
encoder='efficientnet-b4'
pretrained=False
weights=[]
in_channels=4
out_channels=out_channels
initial_lr=0.001
final_lr=0.0005
plot_calib=False
ttas = ['vflip']

### Trainer parameters
max_steps=50000
accelerator='cpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches = 1.
limit_val_batches = 1.
#save_dir='/scratchl/pfournie/outputs/digitaniev2'
save_dir='/work/OT/ai4usr/fournip/outputs/digitanie'
#save_dir='/data/outputs/digitaniev2'
#save_dir='/home/pfournie/ai4geo/ouputs/semcity'

### Building training and validation datasets to concatenate from the splitfile lines
val_sets, train_sets = [], []
data_path = Path(data_path)
splitfile_path=Path.home() / f'dl_toolbox/dl_toolbox/lightning_datamodules/splits/digitanie/{split}.csv'
dataset_factory = DatasetFactory()
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
                crop_size=crop_size,
                fixed_crops=True if is_val else False,
                img_aug=None if is_val else img_aug,
                labels=labels,
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
    batch_size=batch_size,
    collate_fn=CustomCollate(),
    sampler=RandomSampler(
        data_source=train_set,
        replacement=True,
        num_samples=epoch_len
    ),
    num_workers=workers,
    pin_memory=True,
    drop_last=True
)
val_set = ConcatDataset(val_sets)
val_dataloader = DataLoader(
    dataset=val_set,
    shuffle=False,
    collate_fn=CustomCollate(),
    batch_size=batch_size,
    num_workers=workers,
    pin_memory=True
)

### Building lightning module
module = CE(
    ignore_zero=ignore_zero,
    #no_pred_zero=True,
    #mixup=0.4,
    #network='Vgg',
    network=network,
    encoder=encoder,
    pretrained=pretrained,
    weights=weights,
    in_channels=in_channels,
    out_channels=out_channels,
    initial_lr=initial_lr,
    final_lr=final_lr,
    plot_calib=plot_calib,
    class_names=list(val_set.datasets[0].labels.keys()),
    ttas=ttas
    #alphas=(0., 1.),
    #ramp=(0, 40000),
    #pseudo_threshold=0.9,
    #consist_aug='color-3',
    #emas=(0.9, 0.999)
)

### Trainer instance
trainer = Trainer(
    max_steps=max_steps,
    accelerator=accelerator,
    devices=devices,
    multiple_trainloader_mode=multiple_trainloader_mode,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    logger=TensorBoardLogger(
        save_dir=save_dir,
        name=split,
        version=f'{datetime.now():%d%b%y-%Hh%Mm%S}'
    ),
    callbacks=[
        ModelCheckpoint(),
    ]
)
    
#ckpt_path='/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'
trainer.fit(
    model=module,
    train_dataloaders=train_dataloaders,
    val_dataloaders=val_dataloader,
    ckpt_path=None
)
