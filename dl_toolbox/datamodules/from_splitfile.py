from argparse import ArgumentParser
from pathlib import Path
import csv
import ast 
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window
import dl_toolbox.datasets as datasets


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

            ds_name, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
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
                    labels=labels,
                    mins=ast.literal_eval(mins),
                    maxs=ast.literal_eval(maxs)
                )
                sets.append(ds)
    
    return sets

def gen_dataset_args_from_splitfile(
    splitfile_path,
    data_path,
    folds,
):
    
    with open(splitfile_path, newline='') as splitfile:
        reader = csv.reader(splitfile)
        next(reader)
        for row in reader:
            args = {}
            class_name, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
            if int(fold) in folds:
                window = Window(
                    col_off=int(x0),
                    row_off=int(y0),
                    width=int(w),
                    height=int(h)
                )
                args['tile'] = window
                args['image_path'] = data_path/image_path
                args['label_path'] = data_path/label_path if label_path else None
                args['mins'] = ast.literal_eval(mins)
                args['maxs'] = ast.literal_eval(maxs)
                yield class_name, args.copy()
                
dataset_factory = datasets.DatasetFactory()

class FromSplitfile(LightningDataModule):

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
        self.splitfile_path = splitfile_path
        self.data_path = data_path
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.crop_size = crop_size

    def setup(self, stage):
        
        train_sets = []
        for ds_name, args in gen_dataset_args_from_splitfile(
            self.splitfile_path,
            self.data_path,
            self.train_folds
        ):
            ds = dataset_factory.create(ds_name)(
                crop_size=crop_size,
                fixed_crops=False,
                img_aug=img_aug,
                labels=labels,
                bands=bands,
                **args
            )
            train_sets.append(ds)
        self.train_set = ConcatDataset(train_sets)
            
        val_sets = []
        for ds_name, args in gen_dataset_args_from_splitfile(
            splitfile_path,
            data_path,
            val_folds
        ):
            ds = dataset_factory.create(ds_name)(
                crop_size=crop_size,
                fixed_crops=True,
                img_aug=None,
                labels=labels,
                bands=bands,
                **args
            )
            val_sets.append(ds)
        self.val_set = ConcatDataset(test_sets)

        unsup_data = True
        if unsup_data:
            unsup_train_sets = []
            for ds_name, args in gen_dataset_args_from_splitfile(
                splitfile_path,
                data_path,
                list(range(10))
            ):
                ds = dataset_factory.create(ds_name)(
                    crop_size=crop_size,
                    fixed_crops=False,
                    img_aug=img_aug,
                    labels=labels,
                    bands=bands,
                    **args
                )
                unsup_train_sets.append(ds)
            self.unsup_train_set = ConcatDataset(unsup_train_sets)
        
        self.class_names = list(self.val_set.datasets[0].labels.keys())
    
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