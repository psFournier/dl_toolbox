import pytorch_lightning as pl
import os
import csv
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from pathlib import Path
from datetime import datetime

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils

from dl_toolbox.datasets import Raster
from rasterio.windows import from_bounds


if os.uname().nodename == 'WDTIS890Z': 
    data_root = Path('/mnt/d/pfournie/Documents/data')
    home = Path('/home/pfournie')
    save_root = data_root / 'outputs'
elif os.uname().nodename == 'qdtis056z': 
    data_root = Path('/data')
    home = Path('/d/pfournie')
    save_root = data_root / 'outputs'
else:
    #data_root = Path('/work/OT/ai4geo/DATA/DATASETS')
    data_root = Path(os.environ['TMPDIR'])
    home = Path('/home/eh/fournip')
    save_root = Path('/work/OT/ai4usr/fournip') / 'outputs'

# datasets params
data_path = data_root / 'DIGITANIE'
split = home / f'dl_toolbox/dl_toolbox/datamodules/digitanie_toulouse.csv'
train_idx = [0,1,2,3,4,5]
val_idx = [6,7]
crop_size=256
crop_step=256
aug = 'd4_color-3'
val_aug = 'd4'
labels = 'building'
bands = [1,2,3]
unsup_crop_size = 256
unsup_aug = 'd4'
unsup_idx = [10]

# dataloaders params
batch_size = 16
epoch_steps = 500
num_samples = epoch_steps * batch_size
num_workers=6

# module params
mixup=0. # incompatible with ignore_zero=True
weights = [5.,1.]
network='SmpUnet'
encoder='efficientnet-b1'
pretrained=False
in_channels=len(bands)
out_channels=2
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(2,4,0.,0.)
pseudo_threshold=0.9
consist_aug='color-5'
ema_ramp=utils.SigmoidRamp(2,4,0.9,0.99)

# trainer params
num_epochs = 50
max_steps=num_epochs * epoch_steps
accelerator='gpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches=1.
limit_val_batches=1.
save_dir = save_root / 'DIGITANIE'
log_name = 'toulouse'
ckpt_path=None # '/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'

# other
nomenclature = datasets.Digitanie.nomenclatures[labels].value
class_names = [label.name for label in nomenclature.labels]
class_num = len(nomenclature.labels)

for _ in range(5):
    
    train_idx = list(np.random.choice(8, 6, replace=False))
    val_idx = [i for i in range(8) if i not in train_idx]
    print(train_idx, val_idx)
    
    train_sets = datasets.datasets_from_csv(data_path, split, train_idx)
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
    
    unsup_sets = []

    train_set_bounds = []
    for ds in train_sets:
        window = ds.raster.window
        train_set_bounds.append(rasterio.windows.bounds(window, ds.raster_tf))

    for ds in datasets.datasets_from_csv(data_path, split, unsup_idx):

        tf = ds.get_transform()
        unsup_sets.append(
            Raster(
                raster=ds,
                crop_size=unsup_crop_size,
                aug=unsup_aug,
                bands=bands,
                labels=labels,
                holes=[from_bounds(*b, tf) for b in train_set_bounds]
            )
        )

    unsup_set = ConcatDataset(unsup_sets) 

    unsup_dataloader = DataLoader(
        dataset=unsup_set,
        batch_size=batch_size,
        collate_fn=collate.CustomCollate(),
        sampler=RandomSampler(
            data_source=unsup_set,
            replacement=True,
            num_samples=num_samples
        ),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_dataloaders['unsup'] = unsup_dataloader

    val_sets = datasets.datasets_from_csv(data_path, split, val_idx)
    val_set = ConcatDataset(
        [
            datasets.Raster(
                raster=ds,
                crop_size=crop_size,
                aug=val_aug,
                bands=bands,
                labels=labels
            ) for ds in val_sets
        ]
    )
    val_dataloader = DataLoader(
        dataset=val_set,
        sampler=RandomSampler(
            data_source=val_set,
            replacement=True,
            num_samples=num_samples//10
        ),
        collate_fn=collate.CustomCollate(),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    ### Building lightning module
    module = modules.Supervised(
        mixup=mixup, # incompatible with ignore_zero=True
        network=network,
        encoder=encoder,
        pretrained=pretrained,
        weights=weights,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_lr=initial_lr,
        ttas=ttas,
        #alpha_ramp=alpha_ramp,
        #pseudo_threshold=pseudo_threshold,
        #consist_aug=consist_aug,
        #ema_ramp=ema_ramp
    )

    ### Metrics and plots from confmat callback
    metrics_from_confmat = callbacks.MetricsFromConfmat(        
        num_classes=class_num,
        class_names=class_names
    )

    ### Trainer instance
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        multiple_trainloader_mode=multiple_trainloader_mode,
        num_sanity_val_steps=0,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=save_dir,
            name=log_name,
            version=f'{datetime.now():%d%b%y-%Hh%Mm%S}'
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(),
            pl.callbacks.EarlyStopping(monitor='Val_loss', patience=10),
            metrics_from_confmat
        ]
    )

    trainer.fit(
        model=module,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path
    )
