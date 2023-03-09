from argparse import ArgumentParser
import os
import csv

from dl_toolbox.lightning_datamodules import SemisupDm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

from dl_toolbox.utils import worker_init_function 
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *


class SplitfileSemisup(SemisupDm):

    def __init__(
        self,
        data_path,
        splitfile_path,
        test_folds,
        train_folds,
        unsup_splitfile_path,
        unsup_train_folds,
        crop_size,
        img_aug,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.splitfile_path = splitfile_path
        self.test_folds = test_folds
        self.train_folds = train_folds
        self.unsup_splitfile_path = unsup_splitfile_path
        self.unsup_train_folds = unsup_train_folds
        self.crop_size = crop_size
        self.data_path = data_path
        self.img_aug = img_aug

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--unsup_splitfile_path", type=str)
        parser.add_argument("--unsup_train_folds", nargs='+', type=int)
        parser.add_argument("--splitfile_path", type=str)
        parser.add_argument("--test_folds", nargs='+', type=int)
        parser.add_argument("--train_folds", nargs='+', type=int)
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--crop_size", type=int)
        parser.add_argument("--img_aug", type=str)

        return parser

    def setup(self, stage=None):

        with open(self.splitfile_path, newline='') as splitfile:
            train_args, val_args = read_splitfile(
                splitfile=splitfile,
                data_path=self.data_path,
                train_folds=self.train_folds,
                test_folds=self.test_folds
            )

        if train_args:
            self.train_set = ConcatDataset([
                cls(
                    labels=self.labels,
                    img_aug=self.img_aug,
                    crop_size=self.crop_size,
                    crop_step=self.crop_size,
                    one_hot=False,
                    **kwarg
                ) for cls, kwarg in train_args
            ])

        if val_args:
            self.val_set = ConcatDataset([
                cls(
                    labels=self.labels,
                    img_aug='no',
                    crop_size=self.crop_size,
                    crop_step=self.crop_size,
                    one_hot=False,
                    **kwarg
                ) for cls, kwarg in val_args
            ])

        with open(self.unsup_splitfile_path, newline='') as splitfile:
            unsup_train_args, _ = read_splitfile(
                splitfile=splitfile,
                data_path=self.data_path,
                train_folds=self.unsup_train_folds,
                test_folds=()
            )

        if unsup_train_args:
            self.unsup_train_set = ConcatDataset([
                cls(
                    labels=self.labels,
                    img_aug=self.img_aug,
                    crop_size=self.crop_size,
                    crop_step=self.crop_size,
                    one_hot=False,
                    **kwarg
                ) for cls, kwarg in unsup_train_args
            ])


def main():

    datamodule = FromSplitfile(
        dataset_cls=SemcityBdsdDs,
        data_path='/d/pfournie/ai4geo/data/SemcityTLS_DL',
        splitfile_path='/d/pfournie/ai4geo/split_semcity.csv',
        test_folds=(4,),
        train_folds=(0,1,2,3),
        crop_size=128,
        epoch_len=100,
        sup_batch_size=16,
        workers=0,
        img_aug='d4_color-0',
        batch_aug='no',
    )

    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:

        print(batch['image'].shape)
        print(batch['mask'].shape)

if __name__ == '__main__':

    main()

