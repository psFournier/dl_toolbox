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
from .digitanie_ai4geo import DigitanieAi4geo
from dl_toolbox.transforms import NoOp


def is_window_in_poly(w, tf, p):
    bbox = W.bounds(w, transform=tf)
    w_poly = shapely.geometry.box(*bbox)
    return w_poly.within(p)

class Digitanie(DigitanieAi4geo):

    def __init__(
        self,
        city,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.city = city.upper()
        
    def prepare_data(self):
        citypath = self.data_path/f'DIGITANIE_v4/{self.city}'
        imgs = list(citypath.glob('*16bits_COG_*.tif'))
        imgs = sorted(imgs, key=lambda x: int(x.stem.split('_')[-1]))
        msks = list(citypath.glob('COS43/*[0-9].tif'))
        msks = sorted(msks, key=lambda x: int(x.stem.split('_')[-1]))
        pairs = list(zip(imgs,msks))
        self.test = [(i,m,w) for i,m in pairs[:2] for w in get_tiles(2048,2048,512,step_w=256)]
        self.val = [(i,m,w) for i,m in pairs[2:3] for w in get_tiles(2048,2048,512,step_w=256)]
        self.train_s = [(i,m,w) for i,m in pairs[3:] for w in get_tiles(2048,2048,512)][::self.sup]
        if self.unsup != -1: 
            self.toa = next(citypath.glob('*COG.tif'))        
            with rasterio.open(self.toa, 'r') as ds:
                city_tf = ds.transform
                windows = [w[1] for w in ds.block_windows()]
            city_poly = digitanie_polygons[self.city]
            self.toa_windows = [w for w in windows if is_window_in_poly(w,city_tf,city_poly)]
        
    def setup(self, stage):
        self.train_s_set = datasets.Digitanie(
            *[list(t) for t in zip(*self.train_s)],
            self.bands,
            self.merge,
            transforms=self.get_tf(self.train_tf, self.city)
        )
        if self.unsup != -1:
            self.train_u_set = datasets.DigitanieUnlabeledToa(
                toa=self.toa,
                bands=[1,2,3],
                transforms=self.get_tf(NoOp(), self.city),
                windows=self.toa_windows[::self.unsup]
            )
        self.val_set = datasets.Digitanie(
            *[list(t) for t in zip(*self.val)],
            self.bands,
            self.merge,
            transforms=self.get_tf(self.test_tf, self.city)
        )
        self.test_set = datasets.Digitanie(
            *[list(t) for t in zip(*self.test)],
            self.bands,
            self.merge,
            transforms=self.get_tf(self.test_tf, self.city)
        )
                
    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=CustomCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_s_set)(
            sampler=RandomSampler(
                self.train_s_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_s
            ),
            drop_last=True,
            batch_size=self.batch_size_s
        )
        if self.unsup != -1:
            train_dataloaders["unsup"] = self.dataloader(self.train_u_set)(
                sampler=RandomSampler(
                    self.train_u_set,
                    replacement=True,
                    num_samples=self.steps_per_epoch*self.batch_size_u
                ),
                drop_last=True,
                batch_size=self.batch_size_u
            )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
    
    def test_dataloader(self):
        return self.dataloader(self.test_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
