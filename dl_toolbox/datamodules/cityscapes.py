from pathlib import Path

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

import dl_toolbox.datasets as datasets


class Cityscapes(LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers, train_tf, val_tf):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.quality_mode = "fine"
        self.target_type = "semantic"
        self.train_tf = train_tf
        self.val_tf = val_tf
        self.classes = datasets.Cityscapes.classes

    @property
    def in_channels(self):
        return 3

    @property
    def class_colors(self):
        return [(255, (0, 0, 0))] + [
            (l.train_id, l.color)
            for l in datasets.Cityscapes.classes
            if not l.ignore_in_eval
        ]

    @property
    def num_classes(self):
        return 20

    @property
    def class_names(self):
        return ["ignore"] + [
            l.name for l in datasets.Cityscapes.classes if not l.ignore_in_eval
        ]

    def setup(self, stage):
        self.train_set = datasets.Cityscapes(
            self.data_path,
            split="train",
            target_type=self.target_type,
            mode=self.quality_mode,
            transforms=self.train_tf,
        )

        self.val_set = datasets.Cityscapes(
            self.data_path,
            split="val",
            target_type=self.target_type,
            mode=self.quality_mode,
            transforms=self.val_tf,
        )

    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return CombinedLoader(train_dataloaders, mode="min_size")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
