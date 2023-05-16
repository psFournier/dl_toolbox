import pytorch_lightning as pl
import os
import numpy as np
import torch
import math
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, RandomSampler, ConcatDataset
from pytorch_lightning.utilities import CombinedLoader
from pathlib import Path
from datetime import datetime
import torchvision.models as models

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils
import dl_toolbox.torch_sample as sample

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
        data_root = Path(os.environ['TMPDIR'])

# datasets params
dataset_name = 'NWPU-RESISC45'
data_path = data_root / dataset_name
nomenclature = datasets.ResiscNomenclatures['all'].value
num_classes=len(nomenclature)

train = (0,50)
train_idx = [700*i+j for i in range(num_classes) for j in range(*train)]
train_aug = 'd4_color-3'

val = (550, 600)
val_idx = [700*i+j for i in range(num_classes) for j in range(*val)]
val_aug = 'd4_color-3'

unsup_train = (0, 700)
unsup_idx = [700*i+j for i in range(num_classes) for j in range(*unsup_train)]
unsup_aug = 'd4'

# dataloaders params
batch_size = 16
epoch_steps = 200
num_samples = epoch_steps * batch_size
num_workers=6

# network params
in_channels=3
out_channels=num_classes
pretrained = weights = 'IMAGENET1K_V1'

# module params
mixup=0. # incompatible with ignore_zero=True
class_weights = [1.] * num_classes
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(25,50,0.,10.)
pseudo_threshold=0.99
consist_aug='color-5'

# trainer params
num_epochs = 200
#max_steps=num_epochs * epoch_steps
accelerator='gpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches=1.
limit_val_batches=1.
save_dir = save_root / dataset_name
log_name = 'cps_tr50_thr99_ramp10_color5'
ckpt_path=None 


train_set = Subset(
    datasets.Resisc(
        data_path=data_path,
        img_aug=train_aug,
        nomenclature=nomenclature
    ),
    indices=train_idx
)

val_set = Subset(
    datasets.Resisc(
        data_path=data_path,
        img_aug=val_aug,
        nomenclature=nomenclature
    ),
    indices=val_idx
)   

unsup_set = Subset(
    datasets.Resisc(
        data_path=data_path,
        img_aug=unsup_aug,
        nomenclature=nomenclature
    ),
    indices=unsup_idx
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
    drop_last=True
)

train_dataloaders['unsup'] = DataLoader(
    dataset=unsup_set,
    batch_size=batch_size,
    collate_fn=collate.CustomCollate(),
    sampler=RandomSampler(
        data_source=unsup_set,
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


from torchvision.ops.misc import Conv2dNormActivation
from torch import nn

network = models.efficientnet_b0(
    num_classes=out_channels if pretrained is None else 1000,
    weights=pretrained
)
if weights is not None:
    # switching head for num_class and init
    head = nn.Linear(1280, out_channels) # 1280 comes from 4 * lastconv_input_channels=320 in efficientnet_b0
    network.classifier[-1] = head
    init_range = 1.0 / math.sqrt(out_channels)
    nn.init.uniform_(head.weight, -init_range, init_range)
    nn.init.zeros_(head.bias)
    
network2 = models.efficientnet_b0(
    num_classes=out_channels if pretrained is None else 1000,
    weights=pretrained
)
if weights is not None:
    # switching head for num_class and init
    head2 = nn.Linear(1280, out_channels) # 1280 comes from 4 * lastconv_input_channels=320 in efficientnet_b0
    network2.classifier[-1] = head2
    init_range = 1.0 / math.sqrt(out_channels)
    nn.init.uniform_(head2.weight, -init_range, init_range)
    nn.init.zeros_(head2.bias)


### Building lightning module
module = modules.CrossPseudoSupervision(
    mixup=mixup, # incompatible with ignore_zero=True
    network=network,
    num_classes=num_classes,
    class_weights=class_weights,
    initial_lr=initial_lr,
    ttas=ttas,
    alpha_ramp=alpha_ramp,
    pseudo_threshold=pseudo_threshold,
    consist_aug=consist_aug,
    network2=network2
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
        #    patience=10
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

