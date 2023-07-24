from argparse import ArgumentParser
from pathlib import Path
import csv
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window
from dl_toolbox.torch_datasets import *


class FromFolderDataset(LightningDataModule):
    def __init__(
        self,
        folder_dataset,
        data_path,
        batch_size,
        workers,
        train_idxs,
        test_idxs,
        unsup_train_idxs=None,
        img_aug=None,
        unsup_img_aug=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = workers
        dataset_factory = DatasetFactory()

        self.train_set = Subset(
            dataset=dataset_factory.create(folder_dataset)(
                data_path=data_path,
                img_aug=img_aug,
            ),
            indices=train_idxs
            # indices=[700*i+j for i in range(45) for j in range(50)]
        )

        self.val_set = Subset(
            dataset=dataset_factory.create(folder_dataset)(
                data_path=data_path, img_aug="no"
            ),
            indices=test_idxs,
        )

        self.class_names = self.val_set.dataset.class_names

        if unsup_train_idxs:
            self.unsup_train_set = Subset(
                dataset=dataset_factory.create(folder_dataset)(
                    data_path=data_path, img_aug=unsup_img_aug
                ),
                indices=unsup_train_idxs,
            )
        else:
            self.unsup_train_set = None

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--folder_dataset", type=str)
        parser.add_argument("--test_idxs", nargs="+", type=int)
        parser.add_argument("--train_idxs", nargs="+", type=int)
        parser.add_argument("--unsup_train_idxs", nargs="+", type=int)
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--img_aug", type=str)
        parser.add_argument("--unsup_img_aug", type=str)

        return parser

    def train_dataloader(self):
        train_dataloaders = {}

        train_dataloaders["sup"] = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=CustomCollate(),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.unsup_train_set:
            train_dataloaders["unsup"] = DataLoader(
                dataset=self.unsup_train_set,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=CustomCollate(),
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

        return train_dataloaders

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.nb_val_batch = len(self.val_set) // self.batch_size

        return val_dataloader
