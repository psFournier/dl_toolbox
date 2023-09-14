import ast
import csv

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from functools import partial
import numpy as np
import pandas as pd
from itertools import product
import rasterio
import rasterio.windows as windows

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate, get_tiles


class DigitanieAi4geo(LightningDataModule):
    
    cities = {
        'CAN-THO': '6_2',
        'HELSINKI': '8_1',
        'MAROS': '0_2',
        'ARCACHON': '0_1',
        'PARIS': '5_3',
        'SAN-FRANCISCO': '7_9',
        'SHANGHAI': '8_7',
        'MONTPELLIER': '2_0',
        'TOULOUSE': '5_2',
        'PARIS-NEW': '15_19',
        'NEW-YORK': '7_3',
        'NANTES': '2_9',
        'TIANJIN': '4_8',
        'STRASBOURG': '1_9',
        'BIARRITZ': '5_7',
        'BRISBANE': '9_8',
        'BUENOS-AIRES': '0_5',
        'LAGOS': '9_4',
        'LE-CAIRE': '2_6',
        'MUNICH': '5_1',
        'PORT-ELISABETH': '6_9',
        'RIO-JANEIRO': '8_9'
    }

    def __init__(
        self,
        data_path,
        merge,
        bands,
        dataset_tf,
        batch_size,
        num_workers,
        pin_memory,
        class_weights=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.bands = bands
        self.dataset_tf = dataset_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.classes = datasets.Digitanie.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
        
    def prepare_data(self):
        self.dict_train = {'IMG':[], 'MSK':[], "WIN":[]}
        self.dict_val = {'IMG':[], 'MSK':[], "WIN": []}
        for city, val_test in self.cities.items():
            citypath = self.data_path/f'DIGITANIE_v4/{city}'
            val_idx, test_idx = map(int, val_test.split('_'))
            imgs = list(citypath.glob('*16bits_COG_*.tif'))
            imgs = sorted(imgs, key=lambda x: int(x.stem.split('_')[-1]))
            msks = list(citypath.glob('COS9/*.tif'))
            msks = sorted(msks, key=lambda x: int(x.stem.split('_')[-2]))
            try:
                assert len(msks)==len(imgs)
            except AssertionError:
                print(city, ' is not ok')
            nums = range(len(imgs))
            windows = get_tiles(2048, 2048, 512)
            for prod in product(windows, zip(imgs, msks, nums)):
                win, (img, msk, num) = prod
                if num == val_idx:
                    self.dict_val['IMG'].append(img)
                    self.dict_val['MSK'].append(msk)
                    self.dict_val['WIN'].append(win)
                elif num == test_idx:
                    pass
                else:
                    self.dict_train['IMG'].append(img)
                    self.dict_train['MSK'].append(msk)
                    self.dict_train['WIN'].append(win)
        
    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_set = datasets.Digitanie(
                self.dict_train["IMG"],
                self.dict_train["MSK"],
                self.dict_train["WIN"],
                self.bands,
                self.merge,
                transforms=self.dataset_tf,
            )
            self.val_set = datasets.Digitanie(
                self.dict_val["IMG"],
                self.dict_val["MSK"],
                self.dict_val["WIN"],
                self.bands,
                self.merge,
                transforms=self.dataset_tf,
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
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )