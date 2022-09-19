from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate
import torch

from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *

class ResiscDm(LightningDataModule):

    def __init__(
        self,
        epoch_len,
        sup_batch_size,
        workers,
        img_aug,
        batch_aug,
        *args,
        **kwargs
    ):

        super().__init__()
        self.epoch_len = epoch_len
        self.sup_batch_size = sup_batch_size
        self.num_workers = workers
        self.sup_train_set = None
        self.val_set = None
        self.img_aug = img_aug
        self.batch_aug = batch_aug

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--epoch_len", type=int, default=10000)
        parser.add_argument("--sup_batch_size", type=int, default=16)
        parser.add_argument("--workers", default=6, type=int)
        parser.add_argument('--img_aug', type=str, default='no')
        parser.add_argument('--batch_aug', type=str, default='no')

        return parser
    
    def setup(self, stage=None):

        self.sup_train_set = ResiscDs(
            datadir='/d/pfournie/ai4geo/data/NWPU-RESISC45',
            idxs=list(range(1, 650)),
            img_aug=self.img_aug
        )

        self.val_set = ResiscDs(
            datadir='/d/pfournie/ai4geo/data/NWPU-RESISC45',
            idxs=list(range(650, 700)),
            img_aug='no'
        )

    def train_dataloader(self):

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.sup_batch_size,
            collate_fn=CustomCollate(self.batch_aug),
            sampler=sup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function,
            drop_last=True
        )

        return sup_train_dataloader

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

        return val_dataloader
    
def main():

    datamodule = ResiscDm(
        epoch_len=10,
        sup_batch_size=4,
        workers=0,
        img_aug='no',
        batch_aug='no'
    )

    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:

        print(batch['image'].shape)
        print(batch['mask'].shape)

if __name__ == '__main__':

    main()
