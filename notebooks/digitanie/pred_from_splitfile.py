import pytorch_lightning as pl
import os
import csv

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
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
    
# datasets params
data_path = data_root / 'DIGITANIE'
split_dir = home / f'dl_toolbox/dl_toolbox/datamodules'
test_split = (split_dir/'digitanie_toulouse.csv', [8,9])
crop_size=256
crop_step=256
labels = 'mainFuseVege'
bands = [1,2,3]
# dataloaders params
batch_size = 16
num_workers=6
# module params
mixup=0. # incompatible with ignore_zero=True
weights = [1,1,1,1,1,1,]
network='SmpUnet'
encoder='efficientnet-b0'
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
accelerator='gpu'
devices=1
save_dir = save_root / 'DIGITANIE'
log_name = 'toulouse'
version = '14Mar23-19h57m43'
ckpt_path=save_dir/log_name/version/'checkpoints'/'epoch=4-step=5.ckpt'

test_sets = datasets.datasets_from_csv(data_path, *test_split)
test_set = ConcatDataset(
    [datasets.PretiledRaster(ds, crop_size, crop_step, aug=None, bands=bands, labels=labels) for ds in test_sets]
)
test_dataloader = DataLoader(
    dataset=test_set,
    shuffle=False,
    collate_fn=collate.CustomCollate(),
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True
)

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

prediction_writer = callbacks.TiffPredictionWriter(        
    out_path=save_dir/log_name/version/'preds',
    write_interval="batch",
    mode='probas'
)

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    logger=None,
    callbacks=[prediction_writer]
)
    
trainer.predict(
    model=module,
    dataloaders=test_dataloader,
    ckpt_path=ckpt_path,
    return_predictions=False
)
