import ast
import csv

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import rasterio
import rasterio.windows as windows

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler
from torchvision.transforms import transforms

import dl_toolbox.datasets as datasets
import dl_toolbox.datasources as datasources
import dl_toolbox.transforms as tfs

from dl_toolbox.datasets import Raster
from dl_toolbox.utils import CustomCollate


def splits_from_csv(datasrc, datapath, csvpath):
    splits = [[], [], []]

    df_split = pd.read_csv(csvpath / "split.csv", index_col=0)
    df_stats = pd.read_csv(csvpath / "stats.csv", index_col=0)
    df_cls = pd.read_csv(csvpath / "cls.csv", index_col=0)

    for index, row in df_split.iterrows():
        stats = {}
        for p in [0, 0.5, 1, 2, 98, 99, 99.5, 100]:
            stats[f"p{p}"] = [df_stats.loc[index][f"p{p}_{i}"] for i in range(1, 5)]

        # minval = [df_stats.loc[index][f'p0_{i}'] for i in range(1,5)]
        # maxval = [df_stats.loc[index][f'p100_{i}'] for i in range(1,5)]
        # meanval = [df_stats.loc[index][f'mean_{i}'] for i in range(1,5)]
        # cls_counts = list(df_cls.loc[index][1:])

        splits[row["split"]].append(
            datasrc(
                image_path=datapath / row["img"],
                zone=windows.Window(
                    row["col_off"], row["row_off"], row["width"], row["height"]
                ),
                label_path=datapath / df_cls.loc[index]["mask"],
                stats=stats
                # minval=minval,
                # maxval=maxval,
                # meanval=meanval,
                # cls_counts=cls_counts
            )
        )

    return splits


class Digitanie(LightningDataModule):
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
        pin_memory,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.merge = merge
        self.bands = bands

        self.classes = datasources.Digitanie9.classes[self.merge].value

        self.train_srcs, self.val_srcs, _ = splits_from_csv(
            datasource, Path(data_path), Path(csv_path) / csv_name
        )

        self.train_tf = train_tf
        self.val_tf = val_tf

        self.in_channels = len(self.bands)
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]

    def setup(self, stage):
        self.train_set = ConcatDataset(
            [
                Raster(
                    src,
                    merge=self.merge,
                    bands=self.bands,
                    crop_size=self.crop_size,
                    shuffle=True,
                    transforms=self.train_tf,
                )
                for src in self.train_srcs
            ]
        )

        self.val_set = ConcatDataset(
            [
                Raster(
                    src,
                    merge=self.merge,
                    bands=self.bands,
                    crop_size=self.crop_size,
                    shuffle=False,
                    transforms=self.val_tf,
                    crop_step=self.crop_size,
                )
                for src in self.val_srcs
            ]
        )

    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=CustomCollate(),
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return CombinedLoader(train_dataloaders, mode="min_size")

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class Digitanie2(Digitanie):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage):
        self.train_set = ConcatDataset(
            [
                Raster(
                    src,
                    merge=self.merge,
                    bands=self.bands,
                    crop_size=self.crop_size,
                    shuffle=True,
                    transforms=tfs.Compose(
                        [
                            tfs.StretchToMinmaxBySource(src, self.bands),
                            self.train_tf,
                            # tfs.ZeroAverageBySource(src, self.bands)
                        ]
                    ),
                )
                for src in self.train_srcs
            ]
        )

        self.val_set = ConcatDataset(
            [
                Raster(
                    src,
                    merge=self.merge,
                    bands=self.bands,
                    crop_size=self.crop_size,
                    shuffle=False,
                    transforms=tfs.Compose(
                        [
                            tfs.StretchToMinmaxBySource(src, self.bands),
                            self.val_tf,
                            # tfs.ZeroAverageBySource(src, self.bands)
                        ]
                    ),
                    crop_step=self.crop_size,
                )
                for src in self.val_srcs
            ]
        )
