import pytorch_lightning as pl
import os
import numpy as np
import torch
import math
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from pathlib import Path
from datetime import datetime
import torchvision.models as models

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

# datasets params
dataset_name = 'NWPU-RESISC45'
data_path = data_root / dataset_name

#labels = 'initial'
nomenclature = datasets.ResiscNomenclatures['basket-tennis'].value

train_idx = [700*i+j for i in range(len(nomenclature)) for j in range(0,650)]
val_idx = [700*i+j for i in range(len(nomenclature)) for j in range(650,700)]
crop_size=256
crop_step=256
aug = 'd4_color-5'
val_aug = 'no'


bands = [1,2,3]
unsup_crop_size = 256
unsup_aug = 'd4'
unsup_idx = [700*i+j for i in range(len(nomenclature)) for j in range(700)]

# dataloaders params
batch_size = 16
num_workers=6
epoch_steps = np.ceil(len(train_idx) / batch_size)

# network params
pretrained=False
in_channels=len(bands)
out_channels=len(nomenclature)
weights = 'IMAGENET1K_V1'

# module params
mixup=0. # incompatible with ignore_zero=True
class_weights = [1.] * len(nomenclature)
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(2,4,0.,0.)
pseudo_threshold=0.9
consist_aug='color-5'
ema_ramp=utils.SigmoidRamp(2,4,0.9,0.99)

# trainer params
num_epochs = 100
max_steps=num_epochs * epoch_steps
accelerator='gpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches=1.
limit_val_batches=1.
save_dir = save_root / dataset_name
log_name = 'full'
ckpt_path=None # '/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'

resisc = datasets.Resisc(
    data_path=data_path,
    img_aug=aug,
    nomenclature=nomenclature
)

train_set = Subset(resisc, indices=train_idx)
val_set = Subset(resisc, indices=val_idx)
unsup_train_set = Subset(resisc, indices=unsup_idx)

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

#train_dataloaders['unsup'] = DataLoader(
#    dataset=unsup_train_set,
#    batch_size=batch_size,
#    shuffle=True,
#    collate_fn=collate.CustomCollate(),
#    num_workers=num_workers,
#    pin_memory=True,
#    drop_last=True
#)

val_dataloader = DataLoader(
    dataset=val_set,
    shuffle=False,
    collate_fn=collate.CustomCollate(),
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True
)


network = models.efficientnet_b0(
    num_classes=len(nomenclature) if weights is None else 1000,
    weights=weights
)
if weights is not None:
    # switching head for num_class and init
    head = nn.Linear(1280, len(nomenclature)) # 1280 comes from 4 * lastconv_input_channels=320 in efficientnet_b0
    network.classifier[-1] = head
    init_range = 1.0 / math.sqrt(head.out_features)
    nn.init.uniform_(head.weight, -init_range, init_range)
    nn.init.zeros_(head.bias)
    


### Building lightning module
module = modules.Supervised(
    mixup=mixup, # incompatible with ignore_zero=True
    network=network,
    num_classes=len(nomenclature),
    class_weights=class_weights,
    initial_lr=initial_lr,
    ttas=ttas,
    #alpha_ramp=alpha_ramp,
    #pseudo_threshold=pseudo_threshold,
    #consist_aug=consist_aug,
    #ema_ramp=ema_ramp
)

### Metrics and plots from confmat callback
metrics_from_confmat = callbacks.MetricsFromConfmat(        
    num_classes=len(nomenclature),
    class_names=[label.name for label in nomenclature]
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
