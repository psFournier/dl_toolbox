import random
from pathlib import Path
from functools import partial
import torch

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

import dl_toolbox.datasets as datasets
from dl_toolbox.datamodules.flair.utils import flair_gather_data
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists


class Flair(LightningDataModule):
    
    train_domains = [
        "D004_2021", "D014_2020", "D029_2021", "D031_2019", "D058_2020", "D077_2021", "D067_2021", "D066_2021", "D033_2021", "D055_2018", "D072_2019", "D044_2020", "D017_2018", "D086_2020", "D049_2020", "D016_2020", "D063_2019", "D091_2021", "D070_2020", "D013_2020", "D023_2020", "D074_2020", "D021_2020", "D080_2021", "D078_2021", "D032_2019", "D081_2020", "D046_2019", "D052_2019", "D051_2019", "D038_2021", "D009_2019", "D034_2021", "D006_2020", "D008_2019", "D041_2021", "D035_2020", "D007_2020", "D060_2021", "D030_2021"
    ]
    
    test_domains = [
        "D012_2019", "D022_2021", "D026_2020", "D064_2021", "D068_2021", "D071_2020", "D075_2021", "D076_2019", "D083_2020", "D085_2019"
    ]
    
    def __init__(
        self,
        data_path,
        merge,
        bands,
        train_tf,
        test_tf,
        batch_size,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.bands = bands
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.in_channels = len(self.bands)
        self.class_list = datasets.Flair.all_class_lists[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def setup(self, stage):
        train_domains = [self.data_path/"FLAIR_1/train"/d for d in self.train_domains]
        imgs, msks, mtds = flair_gather_data(
            train_domains, path_metadata=None, use_metadata=False, test_set=False
        )
        flair = partial(datasets.Flair, imgs, msks, self.bands, self.merge)
        l, L = int(0.8*len(imgs)), len(imgs)
        idxs=random.sample(range(L), L)
        self.train_set = Subset(flair(self.train_tf), idxs[:l])
        self.val_set = Subset(flair(self.test_tf), idxs[l:])
        self.predict_set = Subset(flair(self.test_tf), idxs[l:])
                        
    def collate(self, batch):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        batch['target'] = torch.stack(batch['target'])
        return batch
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(
            dataset=self.train_set,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(
            dataset=self.val_set,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate
        )

    def predict_dataloader(self):
        return self.dataloader(
            dataset=self.predict_set,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate
        )