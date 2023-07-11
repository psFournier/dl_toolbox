from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, DataLoader, RandomSampler
from torchvision.transforms import transforms
from pytorch_lightning.utilities import CombinedLoader

import dl_toolbox.datasets as datasets
import dl_toolbox.datasources as datasources
from dl_toolbox.utils import CustomCollate
import dl_toolbox.transforms as tfs 

from pathlib import Path
import pandas as pd

import rasterio
import csv
import numpy as np
import ast 
import rasterio.windows as windows

from dl_toolbox.datasets import Raster


def splits_from_csv(datasrc, datapath, csvpath):
    
    splits = [[],[],[]]

    df_split = pd.read_csv(csvpath/'split.csv', index_col=0)
    df_stats = pd.read_csv(csvpath/'stats.csv', index_col=0)
    df_cls = pd.read_csv(csvpath/'cls.csv', index_col=0)

    for index, row in df_split.iterrows():

        minval = [df_stats.loc[index][f'min_{i}'] for i in range(1,5)]
        maxval = [df_stats.loc[index][f'max_{i}'] for i in range(1,5)]
        meanval = [df_stats.loc[index][f'mean_{i}'] for i in range(1,5)]
        #cls_counts = list(df_cls.loc[index][1:])

        splits[row['split']].append(
            datasrc(
                image_path=datapath/row['img'],
                zone=windows.Window(
                    row['col_off'],
                    row['row_off'],
                    row['width'],
                    row['height']
                ),
                label_path=datapath/df_cls.loc[index]['mask'],
                minval=minval,
                maxval=maxval,
                meanval=meanval,
                #cls_counts=cls_counts
            )
        )
        
    return splits

class SplitFromCsv(LightningDataModule):

    def __init__(
        self,
        datasource,
        merge,
        bands,
        crop_size,
        data_path,
        csv_path,
        csv_name,
        train_tf,
        val_tf,
        batch_size,
        num_workers,
        pin_memory
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.merge = merge
        self.bands = bands
        
        self.train_srcs, self.val_srcs, _ = splits_from_csv(
            datasource,
            Path(data_path),
            Path(csv_path)/csv_name
        )
        
        self.train_tf = train_tf
        self.val_tf = val_tf            
        
    def setup(self, stage):
        
        self.train_set = ConcatDataset([
            Raster(
                src,
                merge=self.merge,
                bands=self.bands,
                crop_size=self.crop_size,
                shuffle=True,
                transforms=self.train_tf
            ) for src in self.train_srcs
        ])
        
        self.val_set = ConcatDataset([
            Raster(
                src,
                merge=self.merge,
                bands=self.bands,
                crop_size=self.crop_size,
                shuffle=False,
                transforms=self.val_tf,
                crop_step=self.crop_size
            ) for src in self.val_srcs
        ])
                    
    @property
    def num_classes(self):
        return len(self.train_srcs[0].classes[self.merge].value)
    
    @property
    def class_names(self):
        return [l.name for l in self.train_srcs[0].classes[self.merge].value]    

    def train_dataloader(self):
        
        train_dataloaders = {}
        train_dataloaders['sup'] = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=CustomCollate(),
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        return CombinedLoader(
            train_dataloaders,
            mode='min_size'
        )

    def val_dataloader(self):
        
        return DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )    
    
class SplitFromCsv2(SplitFromCsv):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        
        super().__init__(*args, **kwargs)
        
    def setup(self, stage):
        
        self.train_set = ConcatDataset([
            Raster(
                src,
                merge=self.merge,
                bands=self.bands,
                crop_size=self.crop_size,
                shuffle=True,
                transforms=tfs.Compose([
                    tfs.StretchToMinmaxBySource(src, self.bands),
                    self.train_tf,
                    tfs.ZeroAverageBySource(src, self.bands)
                ])
            ) for src in self.train_srcs
        ])
        
        self.val_set = ConcatDataset([
            Raster(
                src,
                merge=self.merge,
                bands=self.bands,
                crop_size=self.crop_size,
                shuffle=False,
                transforms=tfs.Compose([
                    tfs.StretchToMinmaxBySource(src, self.bands),
                    self.val_tf,
                    tfs.ZeroAverageBySource(src, self.bands)
                ]),
                crop_step=self.crop_size
            ) for src in self.val_srcs
        ])