import random
from pathlib import Path
from functools import partial

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate

from .utils import flair_gather_data

class Flair(LightningDataModule):
    
    train_domains = ["D004_2021", "D014_2020", "D029_2021", "D031_2019", "D058_2020", "D077_2021", "D067_2021", "D066_2021", "D033_2021", "D055_2018", "D072_2019", "D044_2020", "D017_2018", "D086_2020", "D049_2020", "D016_2020", "D063_2019", "D091_2021", "D070_2020", "D013_2020", "D023_2020", "D074_2020", "D021_2020", "D080_2021", "D078_2021", "D032_2019", "D081_2020", "D046_2019", "D052_2019", "D051_2019", "D038_2021", "D009_2019", "D034_2021", "D006_2020", "D008_2019", "D041_2021", "D035_2020", "D007_2020", "D060_2021", "D030_2021"]
    
    test_domains = ["D012_2019", "D022_2021", "D026_2020", "D064_2021", "D068_2021", "D071_2020", "D075_2021", "D076_2019", "D083_2020", "D085_2019"]
    
    def __init__(
        self,
        data_path,
        merge,
        sup,
        unsup,
        bands,
        dataset_tf,
        batch_size,
        num_workers,
        pin_memory,
        class_weights=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.sup = sup
        self.unsup = unsup
        self.bands = bands
        self.dataset_tf = dataset_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.classes = datasets.Flair.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
        
    def prepare_data(self):
        domains = [self.data_path / "FLAIR_1" / "train" / d for d in self.train_domains]
        all_img, all_msk, all_mtd = flair_gather_data(
            domains, path_metadata=None, use_metadata=False, test_set=False
        )
        self.dict_train = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_train_unlabeled = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_val = {"IMG": [], "MSK": [], "MTD": []}
        for i, (img, msk) in enumerate(zip(all_img, all_msk)):
            if self.sup <= i%100 < self.sup + self.unsup:
                self.dict_train_unlabeled["IMG"].append(img)
            if i%100 < self.sup:
                self.dict_train["IMG"].append(img)
                self.dict_train["MSK"].append(msk)
            elif 90 <= i%100:
                self.dict_val["IMG"].append(img)
                self.dict_val["MSK"].append(msk)

    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_set = datasets.Flair(
                self.dict_train["IMG"],
                self.dict_train["MSK"],
                self.bands,
                self.merge,
                transforms=self.dataset_tf,
            )
            self.val_set = datasets.Flair(
                self.dict_val["IMG"],
                self.dict_val["MSK"],
                self.bands,
                self.merge,
                transforms=self.dataset_tf,
            )
            if self.unsup > 0:
                self.unlabeled_set = datasets.Flair(
                    self.dict_train_unlabeled["IMG"],
                    [],
                    self.bands,
                    self.merge,
                    transforms=self.dataset_tf,
                )
                
    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        if self.unsup > 0:
            train_dataloaders["unsup"] = self.dataloader(self.unlabeled_set)(
                shuffle=True,
                drop_last=True,
            )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )