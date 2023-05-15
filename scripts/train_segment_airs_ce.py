import os
import csv
import numpy as np
import shapely
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from pytorch_lightning.utilities import CombinedLoader
from pathlib import Path
from datetime import datetime

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.networks as networks
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils

import rasterio.windows as windows

torch.set_float32_matmul_precision('high')
test = False
if os.uname().nodename == 'WDTIS890Z': 
    data_root = Path('/mnt/d/pfournie/Documents/data')
    home = Path('/home/pfournie')
    save_root = data_root / 'outputs'
elif os.uname().nodename == 'qdtis056z': 
    data_root = Path('/data')
    home = Path('/d/pfournie')
    save_root = data_root / 'outputs'
elif os.uname().nodename.endswith('sis.cnes.fr'):
    home = Path('/home/eh/fournip')
    save_root = Path('/work/OT/ai4usr/fournip') / 'outputs'
    if test:
        data_root = Path('/work/OT/ai4geo/DATA/DATASETS')
    else:
        #!bash '/home/eh/fournip/dl_toolbox/copy_data_to_node.sh'
        data_root = Path(os.environ['TMPDIR'])

# datasets params
dataset_name = 'miniworld_tif'
data_path = data_root / dataset_name
nomenclature = datasets.AirsNomenclatures['building'].value
num_classes=len(nomenclature)
crop_size=256
crop_step=256
bands = [1,2,3]

# split params
split = home / f'dl_toolbox/dl_toolbox/datamodules/airs_50cm.csv'

train_idx = list(range(100))
train_aug = 'd4_color-3'

val_idx = list(range(625, 639))
val_aug = 'd4_color-3'

# dataloaders params
batch_size = 16
epoch_steps = 200
num_samples = epoch_steps * batch_size
num_workers=6

# network params
in_channels=len(bands)
out_channels=num_classes
pretrained = 'imagenet'
encoder='efficientnet-b3'

# module params
mixup=0. # incompatible with ignore_zero=True
class_weights = [1., 3.] #[1.] * num_classes
initial_lr=0.001
ttas=[]

# trainer params
num_epochs = 200
accelerator='gpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches=1.
limit_val_batches=1.
save_dir = save_root / dataset_name
log_name = 'airs_seg_ce'
ckpt_path='/work/OT/ai4usr/fournip/outputs/miniworld_tif/airs_seg_ce/12May23-14h21m22/checkpoints/epoch=99-step=20000.ckpt' 

train_data_src = [
    src for src in datasets.datasets_from_csv(
        data_path,
        split,
        train_idx
    )
]

train_sets = [
    datasets.Raster(
        data_src=src,
        crop_size=crop_size,
        aug=train_aug,
        bands=bands,
        nomenclature=nomenclature
    ) for src in train_data_src
]

train_set = ConcatDataset(train_sets)

val_data_src = [
    src for src in datasets.datasets_from_csv(
        data_path,
        split,
        val_idx
    )
]

val_sets = [
    datasets.Raster(
        data_src=src,
        crop_size=crop_size,
        #crop_step=crop_step,
        aug=val_aug,
        bands=bands,
        nomenclature=nomenclature
    ) for src in val_data_src
]

val_set = ConcatDataset(val_sets)

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
    drop_last=True
)

val_dataloader = DataLoader(
    dataset=val_set,
    #shuffle=False,
    sampler=RandomSampler(
        data_source=val_set,
        replacement=True,
        num_samples=num_samples//10
    ),
    collate_fn=collate.CustomCollate(),
    batch_size=batch_size,
    num_workers=num_workers
)

network = networks.SmpUnet(
    encoder=encoder,
    in_channels=in_channels,
    out_channels=out_channels,
    pretrained=pretrained
)

module = modules.Supervised(
    mixup=mixup, # incompatible with ignore_zero=True
    network=network,
    num_classes=num_classes,
    class_weights=class_weights,
    initial_lr=initial_lr,
    ttas=ttas
)

### Metrics and plots from confmat callback
metrics_from_confmat = callbacks.MetricsFromConfmat(        
    num_classes=num_classes,
    class_names=[label.name for label in nomenclature]
)

logger = pl.loggers.TensorBoardLogger(
    save_dir=save_dir,
    name=log_name,
    version=f'{datetime.now():%d%b%y-%Hh%Mm%S}'
)

### Trainer instance
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator=accelerator,
    devices=devices,
    num_sanity_val_steps=0,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    logger=logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(),
        #pl.callbacks.EarlyStopping(
        #    monitor='Val_loss',
        #    patience=50
        #),
        metrics_from_confmat,
        callbacks.MyProgressBar()
    ]
)

trainer.fit(
    model=module,
    train_dataloaders=CombinedLoader(
        train_dataloaders,
        mode=multiple_trainloader_mode
    ),
    val_dataloaders=val_dataloader,
    ckpt_path=ckpt_path
)