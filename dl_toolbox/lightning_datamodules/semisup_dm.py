from argparse import ArgumentParser
import os
import csv

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

from dl_toolbox.lightning_datamodules import SupervisedDm
from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *
from .utils import read_splitfile


class SemisupDm(SupervisedDm):

    def __init__(
        self,
        unsup_batch_size,
        *args,
        **kwargs
    ):
        
        super().__init__(*args, **kwargs)
        self.unsup_batch_size = unsup_batch_size

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--unsup_batch_size", type=int, default=16)
        return parser

    def train_dataloader(self):

        train_dataloader = super().train_dataloader()

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.unsup_batch_size,
            sampler=unsup_train_sampler,
            collate_fn=CustomCollate(batch_aug='no'),
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function,
            drop_last=True
        )

        train_dataloaders = {
            "sup": train_dataloader,
            "unsup": unsup_train_dataloader
        }

        return train_dataloaders

def main():

    datamodule = SemisupDm(
        dataset_cls=SemcityBdsdDs,
        data_path='/d/pfournie/ai4geo/data/SemcityTLS_DL',
        splitfile_path='/d/pfournie/ai4geo/split_semcity.csv',
        #data_path='/d/pfournie/ai4geo/data/DIGITANIE',
        #splitfile_path='/d/pfournie/ai4geo/split_toulouse.csv',
        test_folds=(4,),
        train_folds=(0,1,2,3),
        crop_size=128,
        epoch_len=100,
        sup_batch_size=16,
        workers=0,
        img_aug='d4_color-0',
        batch_aug='no',
        unsup_splitfile_path='/d/pfournie/ai4geo/split_semcity_unlabeled.csv',
        unsup_batch_size=8,
        unsup_crop_size=140,
        unsup_train_folds=(0,1,2,3,4)
    )

    datamodule.setup()
    sup_dataloader = datamodule.train_dataloader()['sup']
    unsup_dataloader = datamodule.train_dataloader()['unsup']
    for batch in unsup_dataloader:

        print(batch['image'].shape)

if __name__ == '__main__':

    main()

