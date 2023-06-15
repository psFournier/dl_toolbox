from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, DataLoader, RandomSampler
from torchvision.transforms import transforms
from pytorch_lightning.utilities import CombinedLoader

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate, data_src_from_csv
from pathlib import Path

class FromSplitfile(LightningDataModule):

    def __init__(
        self,
        datasrc,
        dataset,
        data_path,
        split,
        train_idx,
        val_idx,
        val_aug,
        train_aug,
        epoch_len,
        batch_size,
        num_workers,
        pin_memory
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        
        self.train_set = ConcatDataset(
            [
                dataset(
                    data_src=data_src,
                    aug=train_aug,
                ) for data_src in data_src_from_csv(
                    datasrc,
                    Path(data_path),
                    Path(split),
                    train_idx
                )
            ]
        )
        
        self.val_set = ConcatDataset(
            [
                dataset(
                    data_src=data_src,
                    aug=val_aug,
                ) for data_src in data_src_from_csv(
                    datasrc,
                    Path(data_path),
                    Path(split),
                    val_idx
                )
            ]
        )

        self.num_samples = self.hparams.epoch_len * self.hparams.batch_size
        self.num_classes = len(self.train_set[0].nomenclature)
        self.input_dim = len(self.train_set[0].bands)
        self.class_weights = [1.]*self.num_classes


        
    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        pass # load them in init

    def train_dataloader(self):
        
        train_dataloaders = {}
        train_dataloaders['sup'] = DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            collate_fn=CustomCollate(),
            sampler=RandomSampler(
                data_source=self.train_set,
                replacement=True,
                num_samples=self.num_samples
            ),
            num_workers=self.hparams.num_workers,
            drop_last=True
        )
        return CombinedLoader(
            train_dataloaders,
            mode='min_size'
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            sampler=RandomSampler(
                data_source=self.val_set,
                replacement=True,
                num_samples=self.num_samples//10
            ),
            collate_fn=CustomCollate(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass