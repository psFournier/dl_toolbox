from argparse import ArgumentParser
import os
import csv

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

from dl_toolbox.utils import worker_init_function 
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *
from .utils import read_splitfile


class SupervisedDm(LightningDataModule):

    def __init__(self,
                 epoch_len,
                 sup_batch_size,
                 workers,
                 batch_aug,
                 *args,
                 **kwargs):

        super().__init__()
        self.epoch_len = epoch_len
        self.sup_batch_size = sup_batch_size
        self.num_workers = workers
        self.batch_aug = batch_aug

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--epoch_len", type=int)
        parser.add_argument("--sup_batch_size", type=int)
        parser.add_argument("--workers", type=int)
        parser.add_argument('--batch_aug', type=str)

        return parser
   
    def train_dataloader(self):

        train_sampler = RandomSampler(
            data_source=self.train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        train_dataloader = DataLoader(
            dataset=self.train_set,
            batch_size=self.sup_batch_size,
            collate_fn=CustomCollate(batch_aug=self.batch_aug),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function,
            drop_last=True
        )

        return train_dataloader

    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(batch_aug='no'),
            batch_size=self.sup_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )
        self.nb_val_batch = len(self.val_set) // self.sup_batch_size

        return val_dataloader
    

def main():

    datamodule = SupervisedDm(
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
    )

    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:

        print(batch['image'].shape)
        print(batch['mask'].shape)

if __name__ == '__main__':

    main()

