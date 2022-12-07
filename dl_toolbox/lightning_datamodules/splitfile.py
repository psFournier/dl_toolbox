from argparse import ArgumentParser
from pathlib import Path
import csv
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window
from dl_toolbox.torch_datasets import *


def read_splitfile(
    data_path,
    splitfile_path,
    folds,
    fixed_crops,
    crop_size,
    img_aug,
    labels
):
    
    sets = []
    dataset_factory = DatasetFactory()
    
    with open(splitfile_path, newline='') as splitfile:
            
        reader = csv.reader(splitfile)
        next(reader)
        for row in reader:

            ds_name, _, image_path, label_path, x0, y0, w, h, fold = row[:9]
            if int(fold) in folds:
                window = Window(
                    col_off=int(x0),
                    row_off=int(y0),
                    width=int(w),
                    height=int(h)
                )
                ds = dataset_factory.create(ds_name)(
                    image_path=data_path/image_path,
                    label_path=data_path/label_path if label_path else None,
                    tile=window,
                    crop_size=crop_size,
                    fixed_crops=fixed_crops,
                    img_aug=img_aug,
                    labels=labels
                )
                sets.append(ds)
    
    return sets

class Splitfile(LightningDataModule):

    def __init__(
        self,
        epoch_len,
        batch_size,
        workers,
        splitfile_path,
        test_folds,
        train_folds,
        data_path,
        crop_size=256,
        img_aug=None,
        unsup_img_aug=None,
        labels='base',
        unsup_train_folds=None,
        #crop_step=None,
        #one_hot=False,
        *args,
        **kwargs
    ):

        super().__init__()
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.num_workers = workers
        dataset_factory = DatasetFactory()
        test_sets, train_sets = [], []
        data_path = Path(data_path)
        train_sets = read_splitfile(
            data_path,
            splitfile_path,
            folds=train_folds,
            fixed_crops=False,
            crop_size=crop_size,
            img_aug=img_aug,
            labels=labels            
        )
        test_sets = read_splitfile(
            data_path,
            splitfile_path,
            folds=test_folds,
            fixed_crops=True,
            crop_size=crop_size,
            img_aug=None,
            labels=labels            
        )
        self.train_set = ConcatDataset(train_sets)
        self.val_set = ConcatDataset(test_sets)
        self.class_names = self.val_set.datasets[0].labels.keys()
        
        if unsup_train_folds:
            unsup_train_sets = read_splitfile(
                data_path,
                splitfile_path,
                folds=unsup_train_folds,
                fixed_crops=False,
                crop_size=crop_size,
                img_aug=unsup_img_aug,
                labels=labels            
            )
            self.unsup_train_set = ConcatDataset(unsup_train_sets)
        else:
            self.unsup_train_set = None

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--epoch_len", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--splitfile_path", type=str)
        parser.add_argument("--test_folds", nargs='+', type=int)
        parser.add_argument("--train_folds", nargs='+', type=int)
        parser.add_argument("--unsup_train_folds", nargs='+', type=int)
        parser.add_argument("--data_path", type=str)
        parser.add_argument('--img_aug', type=str)
        parser.add_argument('--unsup_img_aug', type=str)
        parser.add_argument('--crop_size', type=int)
        parser.add_argument('--crop_step', type=int)
        parser.add_argument('--labels', type=str)

        return parser
    
    def train_dataloader(self):
        
        train_dataloaders = {}
        train_dataloaders['sup'] = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=CustomCollate(),
            sampler=RandomSampler(
                data_source=self.train_set,
                replacement=True,
                num_samples=self.epoch_len
            ),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        if self.unsup_train_set:
    
            train_dataloaders['unsup'] = DataLoader(
                dataset=self.unsup_train_set,
                batch_size=self.batch_size,
                sampler=RandomSampler(
                    data_source=self.unsup_train_set,
                    replacement=True,
                    num_samples=self.epoch_len
                ),
                collate_fn=CustomCollate(),
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )

        return train_dataloaders

    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.nb_val_batch = len(self.val_set) // self.batch_size

        return val_dataloader