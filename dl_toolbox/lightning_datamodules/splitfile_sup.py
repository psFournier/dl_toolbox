from argparse import ArgumentParser
import os
import csv

from dl_toolbox.lightning_datamodules import SupervisedDm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

from dl_toolbox.utils import worker_init_function 
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *
from .utils import read_splitfile


class SplitfileSup(SupervisedDm):

    def __init__(
        self,
        splitfile_path,
        test_folds,
        train_folds,
        data_path,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        dataset_factory = DatasetFactory()
        with open(splitfile_path, newline='') as splitfile:
            train_args, val_args = read_splitfile(
                splitfile=splitfile,
                data_path=data_path,
                train_folds=train_folds,
                test_folds=test_folds
            )

        if train_args:
            self.train_set = ConcatDataset([
                dataset_factory.create(dataset)(
                    *args,
                    {**kwargs, **read_kwargs}
                ) for dataset, read_kwargs in train_args
            ])

        if val_args:
            self.val_set = ConcatDataset([
                dataset_factory.create(dataset)(
                    *args,
                    {**kwargs, **read_kwargs}
                ) for dataset, kwargs in val_args
            ])

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--splitfile_path", type=str)
        parser.add_argument("--test_folds", nargs='+', type=int)
        parser.add_argument("--train_folds", nargs='+', type=int)
        parser.add_argument("--data_path", type=str)
        parser = RasterDs.add_model_specific_args(parser)

        return parser


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

