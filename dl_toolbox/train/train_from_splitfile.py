from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
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

import dl_toolbox.callbacks as callbacks
import dl_toolbox.lightning_modules as modules 
import dl_toolbox.torch_datasets as datasets
import dl_toolbox.torch_collate as collate
import dl_toolbox.utils as utils



"""
Changer data_path, labels, splitfile_path, epoch_len, save_dir... en fonction de l'expé.
"""


if os.uname().nodename == 'WDTIS890Z': 
    data_root = Path('/mnt/d/pfournie/Documents/data')
    home = Path('/home/pfournie')
elif os.uname().nodename == 'qdtis056z': 
    data_root = Path('/data')
    home = Path('/d/pfournie')
    
def gen_dataset_args_from_splitfile(
    splitfile_path,
    data_path,
    folds,
):
    
    with open(splitfile_path, newline='') as splitfile:
        reader = csv.reader(splitfile)
        next(reader)
        for row in reader:
            args = {}
            _, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
            if int(fold) in folds:
                window = Window(
                    col_off=int(x0),
                    row_off=int(y0),
                    width=int(w),
                    height=int(h)
                )
                args['tile'] = window
                args['image_path'] = data_path/image_path
                args['label_path'] = data_path/label_path if label_path else None
                args['mins'] = ast.literal_eval(mins)
                args['maxs'] = ast.literal_eval(maxs)
                yield args.copy()

data_path = data_root / 'DIGITANIE'
split_name = 'toulouse'
splitfile_path = home / f'dl_toolbox/dl_toolbox/lightning_datamodules/splits/digitanie/{split_name}.csv'
crop_size=256
img_aug = 'd4'
labels = '6mainFuseVege'

for train_folds, val_folds in [([0,1,2,3,4],[5,6])]:
    for mixup in [0]:
        for weights in [[1,1,1,1,1,1]]:
           
            train_sets = []
            for args in gen_dataset_args_from_splitfile(
                splitfile_path,
                data_path,
                train_folds
            ):
                ds = datasets.DigitanieV2(
                    crop_size=crop_size,
                    fixed_crops=False,
                    img_aug=img_aug,
                    labels=labels,
                    **args
                )
                train_sets.append(ds)

            val_sets = []
            for args in gen_dataset_args_from_splitfile(
                splitfile_path,
                data_path,
                val_folds
            ):
                ds = datasets.DigitanieV2(
                    crop_size=crop_size,
                    fixed_crops=True,
                    img_aug=None,
                    labels=labels,
                    **args
                )
                val_sets.append(ds)
                
            unsup_data = True
            if unsup_data:
                unsup_train_sets = []
                for args in gen_dataset_args_from_splitfile(
                    splitfile_path,
                    data_path,
                    list(range(10))
                ):
                    ds = datasets.DigitanieV2(
                        crop_size=crop_size,
                        fixed_crops=False,
                        img_aug=img_aug,
                        labels=labels,
                        **args
                    )
                    unsup_train_sets.append(ds)
                
#            ### Building training and validation datasets to concatenate from the splitfile lines
#            val_sets, train_sets = [], []
#            
#            
#            with open(splitfile_path, newline='') as splitfile:
#                reader = csv.reader(splitfile)
#                next(reader)
#                for row in reader:
#                    ds_name, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
#                    if int(fold) in val_folds+train_folds:
#                        is_val = int(fold) in val_folds
#                        window = Window(
#                            col_off=int(x0),
#                            row_off=int(y0),
#                            width=int(w),
#                            height=int(h)
#                        )
#                        ds = dataset_factory.create(ds_name)(
#                            image_path=data_path/image_path,
#                            label_path=data_path/label_path if label_path else None,
#                            tile=window,
#                            crop_size=256,
#                            fixed_crops=True if is_val else False,
#                            img_aug=None if is_val else 'd4',
#                            labels='6mainFuseVege',
#                            mins=ast.literal_eval(mins),
#                            maxs=ast.literal_eval(maxs)
#                        )
#                        if is_val:
#                            val_sets.append(ds)
#                        else:
#                            train_sets.append(ds)

            ### Building dataloaders
            batch_size = 4
            num_samples = 2000
            num_workers=4
            
            train_set = ConcatDataset(train_sets)
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
            
            unsup_data = True
            if unsup_data:
                unsup_train_set = ConcatDataset(unsup_train_sets)
                train_dataloaders['unsup'] = DataLoader(
                    dataset=unsup_train_set,
                    batch_size=batch_size,
                    collate_fn=collate.CustomCollate(),
                    sampler=RandomSampler(
                        data_source=unsup_train_set,
                        replacement=True,
                        num_samples=num_samples
                    ),
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True
                )
                
            val_set = ConcatDataset(val_sets)
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
                network='SmpUnet',
                encoder='efficientnet-b0',
                pretrained=False,
                weights=weights,
                in_channels=4,
                out_channels=6,
                initial_lr=0.001,
                ttas=[],
                alpha_ramp=utils.SigmoidRamp(2,4,0.,2.),
                pseudo_threshold=0.9,
                consist_aug='d4',
                ema_ramp=utils.SigmoidRamp(3,6,0.9,0.99),
            )

            ### Metrics and plots from confmat callback
            metrics_from_confmat = callbacks.MetricsFromConfmat(        
                num_classes=6,
                class_names=list(val_set.datasets[0].labels.keys())
            )

            ### Trainer instance
            trainer = Trainer(
                max_steps=30000,
                accelerator='gpu',
                devices=1,
                multiple_trainloader_mode='min_size',
                num_sanity_val_steps=0,
                limit_train_batches=1,
                limit_val_batches=1,
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
