from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from dl_toolbox.lightning_datamodules import SupervisedDm
from dl_toolbox.torch_collate import CustomCollate

from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_datasets import *


class SplitIdxSup(SupervisedDm):
    def __init__(self, dataset, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset_factory = DatasetFactory()
        self.train_set = dataset_factory.create(dataset)(
            idxs=tuple(range(1, split[0])), *args, **kwargs
        )
        self.val_set = dataset_factory.create(dataset)(
            idxs=tuple(range(split[0], split[1])), *args, **kwargs
        )
        self.class_names = self.val_set.cls_names

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--split", nargs=2, type=int)
        parser.add_argument("--dataset", type=str)

        return parser


def main():
    datamodule = SplitIdxSup(
        dataset="Resisc",
        img_aug="d4",
        data_path="/home/pfournie/ai4geo/data/NWPU-RESISC45",
        labels="base",
        split=(2, 5),
        epoch_len=10,
        sup_batch_size=4,
        workers=0,
        batch_aug="no",
    )

    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        print(batch["image"].shape)
        print(batch["mask"].shape)


if __name__ == "__main__":
    main()
