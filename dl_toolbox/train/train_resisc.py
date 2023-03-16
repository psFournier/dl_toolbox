import pytorch_lightning as pl
import os
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from pathlib import Path
from datetime import datetime

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils


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
    print('torch device count', torch.cuda.device_count())
    
# datasets params
dataset_name = 'NWPU-RESISC45'
data_path = data_root / dataset_name
aug = 'd4_color-5'
unsup_aug = 'd4'
labels = 'test'
nomenclature = datasets.Resisc.nomenclatures[labels].value
class_names = [label.name for label in nomenclature.labels]
class_num = len(nomenclature.labels)
train_indices = [700*i+j for i in range(class_num) for j in range(0,100)]
unsup_train_indices = [700*i+j for i in range(class_num) for j in range(700)]
val_indices = [700*i+j for i in range(class_num) for j in range(600,650)]

# dataloaders params
batch_size = 16
num_workers=6

# module params
mixup=0. # incompatible with ignore_zero=True
weights = [1,1]
network='Vgg'
pretrained=False
in_channels=3
out_channels=2
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(15,30,0.,1.)
pseudo_threshold=0.9
consist_aug='color-5'
ema_ramp=utils.SigmoidRamp(15,30,0.9,0.99)

# trainer params
num_epochs = 100
max_steps=num_epochs * len(train_indices)
accelerator='gpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches=1.
limit_val_batches=1.
save_dir = save_root / dataset_name
log_name = 'test'
ckpt_path=None # '/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'

resisc = datasets.Resisc(
    data_path=data_path,
    img_aug=aug,
    labels=labels
)

train_set = Subset(resisc, indices=train_indices)
val_set = Subset(resisc, indices=val_indices)
unsup_train_set = Subset(resisc, indices=unsup_train_indices)

train_dataloaders = {}

train_dataloaders['sup'] = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    collate_fn=collate.CustomCollate(),
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

train_dataloaders['unsup'] = DataLoader(
    dataset=unsup_train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate.CustomCollate(),
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
module = modules.MeanTeacher(
    mixup=mixup, # incompatible with ignore_zero=True
    network=network,
    #encoder=encoder,
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
