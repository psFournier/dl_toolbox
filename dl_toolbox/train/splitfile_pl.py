import pytorch_lightning as pl
import os
import csv
import rasterio

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from pathlib import Path
from datetime import datetime

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils

from dl_toolbox.datasets import Raster, PretiledRaster


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
    #data_root = Path(os.environ['TMPDIR'])
    home = Path('/home/eh/fournip')
    save_root = Path('/work/OT/ai4usr/fournip') / 'outputs'

# datasets params
data_path = data_root / 'DIGITANIE'
split_dir = home / f'dl_toolbox/dl_toolbox/datamodules'
train_split = (split_dir/'digitanie_toulouse.csv', [0,1,2,3,4,5])
unsup_train_split = pl_split = (split_dir/'digitanie_toulouse_big_1.csv', [0])
val_split = (split_dir/'digitanie_toulouse.csv', [6,7])
crop_size = 256
pl_crop_size = 10000
crop_step = 256
pl_crop_step = 10000
aug = 'd4_color-5'
unsup_aug = pl_aug = 'd4'
labels = 'mainFuseVege'
bands = [1,2,3]

# dataloaders params
batch_size = 16
epoch_steps = 500
num_samples = epoch_steps * batch_size
num_workers=6

# module params
mixup=0. # incompatible with ignore_zero=True
weights = [1,1,1,1,1,1,]
network='SmpUnet'
encoder='efficientnet-b4'
pretrained=False
in_channels=len(bands)
out_channels=6
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(15,30,0.,1.)
pseudo_threshold=0.9
consist_aug='color-5'
ema_ramp=utils.SigmoidRamp(15,30,0.9,0.99)

# trainer params
num_epochs = 60
max_steps=num_epochs * epoch_steps
accelerator='gpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches=1.
limit_val_batches=1.
save_dir = save_root / 'DIGITANIE'
log_name = 'toulouse'
version = '14Mar23-19h57m43'
ckpt_path=save_dir/log_name/version/'checkpoints'/'epoch=4-step=5.ckpt'

# other
nomenclature = datasets.Digitanie.nomenclatures[labels].value
class_names = [label.name for label in nomenclature.labels]
class_num = len(nomenclature.labels)

train_sets = []
for ds in datasets.datasets_from_csv(data_path, *train_split):
    train_sets.append(Raster(ds, crop_size, aug, bands, labels))
train_set = ConcatDataset(train_sets)
"""
val_sets = []
for ds in datasets.datasets_from_csv(data_path, *val_split):
    val_sets.append(PretiledRaster(ds, crop_size, crop_step, aug=None, bands=bands, labels=labels))
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
    pin_memory=True,
    drop_last=True
)

val_dataloader = DataLoader(
    dataset=val_set,
    shuffle=False,
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
        pl.callbacks.EarlyStopping(monitor='Val_loss', patience=5),
        metrics_from_confmat
    ]
)

trainer.fit(
    model=module,
    train_dataloaders=train_dataloaders,
    val_dataloaders=val_dataloader,
    ckpt_path=ckpt_path
)
"""
pl_sets = []
holes_bounds = []
for ds in train_sets:
    window = ds.raster.window
    tf = ds.raster_tf
    holes_bounds.append(rasterio.windows.bounds(window, tf))
for ds in datasets.datasets_from_csv(data_path, *pl_split):
    with rasterio.open(ds.image_path) as raster_img:
        dst_tf = raster_img.transform
    holes = [rasterio.windows.from_bounds(*b, dst_tf) for b in holes_bounds]
    pl_sets.append(
        PretiledRaster(
            crop_step=pl_crop_step,
            raster=ds,
            crop_size=pl_crop_size,
            aug=pl_aug,
            bands=bands,
            labels=labels,
            holes=holes
        )
    )
    for tile in pl_sets[-1].tiles:
        print(tile)
pl_set = ConcatDataset(pl_sets) 

pl_dataloader = DataLoader(
    dataset=pl_set,
    batch_size=batch_size,
    collate_fn=collate.CustomCollate(),
    sampler=RandomSampler(
        data_source=pl_set,
        replacement=True,
        num_samples=num_samples
    ),
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

prediction_writer = callbacks.TiffPredictionWriter(        
    out_path=save_dir/log_name/version/'preds',
    write_interval="batch",
    mode='probas'
)

pl_trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    logger=None,
    callbacks=[prediction_writer]
)
    
pl_trainer.predict(
    model=module,
    dataloaders=pl_dataloader,
    ckpt_path=ckpt_path,
    return_predictions=False
)
                    
