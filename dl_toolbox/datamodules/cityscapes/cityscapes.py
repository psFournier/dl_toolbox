from pathlib import Path
import os

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

import dl_toolbox.datasets as datasets


class Cityscapes(LightningDataModule):
    def __init__(
        self,
        data_path,
        merge,
        prop,
        train_tf,
        val_tf,
        test_tf,
        batch_size,
        num_workers,
        pin_memory,
        class_weights=None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.prop = prop
        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = 3
        self.classes = datasets.Cityscapes.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
        
    def prepare_data(self):
        def get_split_data(split):
            imgs = []
            msks = []
            img_dir = self.data_path/'Cityscapes'/'leftImg8bit'/split
            msk_dir = self.data_path/'Cityscapes'/'gtFine'/split
            for city in os.listdir(img_dir):
                for file_name in os.listdir(img_dir/city):
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0], "gtFine_labelIds.png"
                    )
                    imgs.append(img_dir/city/file_name)
                    msks.append(msk_dir/city/target_name)
            return {'IMG': imgs, 'MSK': msks}
        self.train_dict = get_split_data("train")
        self.val_dict = get_split_data("val")

    def setup(self, stage):
        self.train_set = datasets.Cityscapes(
            self.train_dict['IMG'],
            self.train_dict['MSK'],
            merge=self.merge,
            transforms=self.train_tf,
        )

        self.val_set = datasets.Cityscapes(
            self.val_dict['IMG'],
            self.val_dict['MSK'],
            merge=self.merge,
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
