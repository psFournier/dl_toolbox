import ast
import csv

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from itertools import product
import rasterio
import rasterio.windows as windows
from functools import partial
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler
import shapely
import rasterio.windows as W

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate, get_tiles
from .digi_polygons import digitanie_polygons
from .digitanie import DigitanieAi4geo


def is_window_in_poly(w, tf, p):
    bbox = W.bounds(w, transform=tf)
    w_poly = shapely.geometry.box(*bbox)
    return w_poly.within(p)

class Digitanie(DigitanieAi4geo):

    def __init__(
        self,
        city,
        sup,
        unsup,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.city = city
        self.sup = sup
        self.unsup = unsup
        
    def prepare_data(self):
        self.dict_train = {'IMG':[], 'MSK':[], "WIN":[]}
        self.dict_val = {'IMG':[], 'MSK':[], "WIN": []}
        citypath = self.data_path/f'DIGITANIE_v4/{self.city}'
        imgs = list(citypath.glob('*16bits_COG_*.tif'))
        imgs = sorted(imgs, key=lambda x: int(x.stem.split('_')[-1]))
        msks = list(citypath.glob('COS9/*.tif'))
        msks = sorted(msks, key=lambda x: int(x.stem.split('_')[-2]))
        try:
            assert len(msks)==len(imgs)
        except AssertionError:
            print(self.city, ' is not ok')
        windows = get_tiles(2048, 2048, 512)
        for i, prod in enumerate(product(windows, zip(imgs, msks))):
            win, (img, msk) = prod
            if i%100 < self.sup:
                self.dict_train['IMG'].append(img)
                self.dict_train['MSK'].append(msk)
                self.dict_train['WIN'].append(win)
            elif i%100 >= 90:
                self.dict_val['IMG'].append(img)
                self.dict_val['MSK'].append(msk)
                self.dict_val['WIN'].append(win)
        if self.unsup>0:
            self.toa = next(citypath.glob('*COG.tif'))
            with rasterio.open(self.toa, 'r') as ds:
                city_tf = ds.transform
                windows = [w[1] for w in ds.block_windows()]
            city_poly = digitanie_polygons[self.city]
            self.toa_windows = [w for w in windows if is_window_in_poly(w,city_tf,city_poly)]
        
    def setup(self, stage):
        super().setup(stage)
        if stage in ("fit", "validate"):
            if self.unsup > 0:
                self.unlabeled_set = datasets.DigitanieUnlabeledToa(
                    toa=self.toa,
                    bands=[1,2,3],
                    transforms=self.dataset_tf,
                    windows=self.toa_windows
                )
                
    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        if self.unsup > 0:
            train_dataloaders["unsup"] = self.dataloader(self.unlabeled_set)(
                shuffle=True,
                drop_last=True,
            )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )
