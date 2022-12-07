from argparse import ArgumentParser
from pathlib import Path
import csv

from dl_toolbox.lightning_datamodules import SupervisedDm
from torch.utils.data import ConcatDataset
from rasterio.windows import Window
from dl_toolbox.torch_datasets import *


class SplitfileSup(SupervisedDm):

    def __init__(
        self,
        splitfile_path,
        test_folds,
        train_folds,
        data_path,
        crop_size,
        img_aug,
        labels,
        #crop_step=None,
        #one_hot=False,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        dataset_factory = DatasetFactory()
        test_sets, train_sets = [], []
        p = Path(data_path)
        
        with open(splitfile_path, newline='') as splitfile:
            
            reader = csv.reader(splitfile)
            next(reader)
            for row in reader:

                ds_name, _, image_path, label_path, x0, y0, w, h, fold = row[:9]
                is_train = int(fold) in train_folds 
                is_test = int(fold) in test_folds
                window = Window(
                    col_off=int(x0),
                    row_off=int(y0),
                    width=int(w),
                    height=int(h)
                )
                if is_train or is_test:
                    ds = dataset_factory.create(ds_name)(
                        image_path=p/image_path,
                        label_path=p/label_path if label_path else None,
                        tile=window,
                        crop_size=crop_size,
                        fixed_crops=is_test,
                        img_aug=img_aug if is_train else None,
                        labels=labels
                    )
                    if is_train:
                        train_sets.append(ds)
                    else:
                        test_sets.append(ds)

        self.train_set = ConcatDataset(train_sets)
        self.val_set = ConcatDataset(test_sets)
        self.class_names = self.val_set.datasets[0].labels.keys()


    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--splitfile_path", type=str)
        parser.add_argument("--test_folds", nargs='+', type=int)
        parser.add_argument("--train_folds", nargs='+', type=int)
        parser.add_argument("--data_path", type=str)
        parser.add_argument('--img_aug', type=str)
        parser.add_argument('--crop_size', type=int)
        parser.add_argument('--crop_step', type=int)
        parser.add_argument('--labels', type=str)

        return parser