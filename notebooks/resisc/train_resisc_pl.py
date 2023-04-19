#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pytorch_lightning as pl
import os
import numpy as np
import torch
import math
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, RandomSampler, ConcatDataset
from pathlib import Path
from datetime import datetime
import torchvision.models as models

import dl_toolbox.callbacks as callbacks
import dl_toolbox.modules as modules 
import dl_toolbox.datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils
import dl_toolbox.torch_sample as sample

from tqdm import tqdm_notebook as tqdm


# In[3]:


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


# In[4]:


# datasets params
dataset_name = 'NWPU-RESISC45'
data_path = data_root / dataset_name
nomenclature = datasets.ResiscNomenclatures['all'].value
num_classes=len(nomenclature)

train = (0,50)
train_idx = [700*i+j for i in range(num_classes) for j in range(*train)]
train_aug = 'd4_color-3'

val = (600, 700)
val_idx = [700*i+j for i in range(num_classes) for j in range(*val)]
val_aug = 'no'
#num_val_samples = 500

unsup_train = (50, 100)
unsup_idx = [700*i+j for i in range(num_classes) for j in range(*unsup_train)]
unsup_aug = 'd4'

# dataloaders params
batch_size = 16
num_workers=6

# network params
out_channels=num_classes
weights = 'IMAGENET1K_V1'

# module params
mixup=0. # incompatible with ignore_zero=True
class_weights = [1.] * num_classes
initial_lr=0.001
ttas=[]
alpha_ramp=utils.SigmoidRamp(2,4,0.,5.)
pseudo_threshold=0.9
consist_aug='color-5'
ema_ramp=utils.SigmoidRamp(2,4,0.9,0.99)

# trainer params
num_epochs = 5
accelerator='gpu'
devices=1
multiple_trainloader_mode='min_size'
limit_train_batches=1.
limit_val_batches=1.
save_dir = save_root / dataset_name
log_name = 'labels:all_nbtran:600'
ckpt_path=None # '/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'


# In[5]:


log_name = f'train={train}_unsup_train={unsup_train}'

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

epoch_steps = int(np.ceil(len(train_set) / batch_size))
num_train_samples = epoch_steps * batch_size
#max_steps=num_epochs * epoch_steps

train_dataloaders = {}

train_dataloaders['sup'] = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    collate_fn=collate.CustomCollate(),
    sampler=RandomSampler(
        data_source=train_set,
        replacement=True,
        num_samples=num_train_samples
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
)


# In[6]:


network = models.efficientnet_b0(
    num_classes=out_channels if weights is None else 1000,
    weights=weights
)
if weights is not None:
    # switching head for num_class and init
    head = nn.Linear(1280, out_channels) # 1280 comes from 4 * lastconv_input_channels=320 in efficientnet_b0
    network.classifier[-1] = head
    init_range = 1.0 / math.sqrt(out_channels)
    nn.init.uniform_(head.weight, -init_range, init_range)
    nn.init.zeros_(head.bias)

### Building lightning module
module = modules.Supervised(
    mixup=mixup, # incompatible with ignore_zero=True
    network=network,
    #network2=network2,
    num_classes=num_classes,
    class_weights=class_weights,
    initial_lr=initial_lr,
    ttas=ttas,
    #alpha_ramp=alpha_ramp,
    #pseudo_threshold=pseudo_threshold,
    #consist_aug=consist_aug,
    #ema_ramp=ema_ramp
)


# In[7]:


### Metrics and plots from confmat callback
metrics_from_confmat = callbacks.MetricsFromConfmat(        
    num_classes=num_classes,
    class_names=[label.name for label in nomenclature]
)

### Trainer instance
logger = pl.loggers.TensorBoardLogger(
    save_dir=save_dir,
    name=log_name,
    version=f'{datetime.now():%d%b%y-%Hh%Mm%S}'
)
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator=accelerator,
    devices=devices,
    multiple_trainloader_mode=multiple_trainloader_mode,
    num_sanity_val_steps=0,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    logger=logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(),
        pl.callbacks.EarlyStopping(
            monitor='Val_loss',
            patience=10
        ),
        metrics_from_confmat,
        callbacks.MyProgressBar()
    ]
)


# In[8]:


trainer.fit(
    model=module,
    train_dataloaders=train_dataloaders,
    val_dataloaders=val_dataloader,
    ckpt_path=ckpt_path
)


# In[9]:


pred_set = Subset(
    datasets.Resisc(
        data_path=data_path,
        img_aug='no',
        nomenclature=nomenclature
    ),
    indices=unsup_idx
)

pred_dataloader = DataLoader(
    dataset=pred_set,
    shuffle=False,
    collate_fn=collate.CustomCollate(),
    batch_size=batch_size,
    num_workers=num_workers
)


# In[10]:


pl_dir = os.path.join(
    save_dir,
    log_name,
    logger.log_dir,
    dataset_name
)

counts = [0] * num_classes

for batch in pred_dataloader:

    logits = module.forward(batch['image'])
    probas = module.logits2probas(logits)
    confs, preds = module.probas2confpreds(probas)

    for i, pred in enumerate(preds):
        if confs[i] > 0.9:
            pred = int(pred)
            cls_name = pred_set.dataset.classes[pred]
            num = counts[pred]
            class_dir = Path(pl_dir) / cls_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = class_dir / f'{cls_name}_{num:04}.jpg'
            os.symlink(
                batch['path'][i],
                dst
            )
            counts[pred] += 1


# In[39]:


pl_set = datasets.Resisc(
    data_path=pl_dir,
    img_aug='no',
    nomenclature=nomenclature
)

pl_train_set = ConcatDataset([train_set, pl_set])

pl_sampler = sample.BalancedConcat(
    lengths=[len(train_set), len(pl_set)],
    num_samples=len(pl_train_set)
)

pl_train_dataloaders = {}

pl_train_dataloaders['sup'] = DataLoader(
    dataset=pl_train_set,
    #sampler=pl_sampler,
    shuffle=True,
    collate_fn=collate.CustomCollate(),
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True
)

#val_set = Subset(
#    datasets.Resisc(
#        data_path=data_path,
#        img_aug=val_aug,
#        nomenclature=nomenclature
#    ),
#    indices=val_idx
#)  
#
#pl_val_dataloader = DataLoader(
#    dataset=val_set,
#    shuffle=False,
#    collate_fn=collate.CustomCollate(),
#    batch_size=batch_size,
#    num_workers=num_workers
#)
#
#print(len(pl_train_set))
#print(num_val_samples)


# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

item = val_set[2280]
fig, ax = plt.subplots(1, 1)
ax.imshow(item['image'].numpy().transpose(1,2,0)[...,:3])
ax.set_title(int(item['label']))
ax.autoscale_view()

for i, batch in enumerate(val_dataloader):
    print(batch.keys())
    for j in range(4):
        f, ax = plt.subplots(ncols=1)
        ax.imshow(batch['image'][j].numpy().transpose(1,2,0)[...,:3])
        ax.set_title(int(batch['label'][j]))

    break


# In[41]:


pl_network = models.efficientnet_b0(
    num_classes=out_channels if weights is None else 1000,
    weights=weights
)
if weights is not None:
    # switching head for num_class and init
    head = nn.Linear(1280, out_channels) # 1280 comes from 4 * lastconv_input_channels=320 in efficientnet_b0
    pl_network.classifier[-1] = head
    init_range = 1.0 / math.sqrt(out_channels)
    nn.init.uniform_(head.weight, -init_range, init_range)
    nn.init.zeros_(head.bias)

pl_module = modules.Supervised(
    mixup=mixup, # incompatible with ignore_zero=True
    network=pl_network,
    num_classes=num_classes,
    class_weights=class_weights,
    initial_lr=initial_lr,
    ttas=ttas
)


# In[42]:


### Trainer instance
pl_trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator=accelerator,
    devices=devices,
    multiple_trainloader_mode=multiple_trainloader_mode,
    num_sanity_val_steps=0,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    logger=logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(),
        pl.callbacks.EarlyStopping(
            monitor='Val_loss',
            patience=10
        ),
        metrics_from_confmat,
        callbacks.MyProgressBar()
    ]
)


# In[43]:


pl_trainer.fit(
    model=pl_module,
    train_dataloaders=pl_train_dataloaders,
    val_dataloaders=val_dataloader
)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')

