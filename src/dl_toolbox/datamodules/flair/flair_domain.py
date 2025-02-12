import random
from pathlib import Path
from functools import partial

from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate

from .utils import flair_gather_data

from .flair import Flair

class FlairDomain(Flair):
    
    train_domains = ["D004_2021", "D014_2020", "D029_2021", "D031_2019", "D058_2020", "D077_2021", "D067_2021", "D066_2021", "D033_2021", "D055_2018", "D072_2019", "D044_2020", "D017_2018", "D086_2020", "D049_2020", "D016_2020", "D063_2019", "D091_2021", "D070_2020", "D013_2020", "D023_2020", "D074_2020", "D021_2020", "D080_2021", "D078_2021", "D032_2019", "D081_2020", "D046_2019", "D052_2019", "D051_2019", "D038_2021", "D009_2019", "D034_2021", "D006_2020", "D008_2019", "D041_2021", "D035_2020", "D007_2020", "D060_2021", "D030_2021"]
    
    test_domains = ["D012_2019", "D022_2021", "D026_2020", "D064_2021", "D068_2021", "D071_2020", "D075_2021", "D076_2019", "D083_2020", "D085_2019"    ]
    
    def __init__(
        self,
        data_path,
        merge,
        prop,
        bands,
        train_tf,
        val_tf,
        test_tf,
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
        self.prop = prop
        self.bands = bands
        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = test_tf
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
        random.shuffle(domains)
        train_domains, val_domains, test_domains = [], [], []
        for i, domain in enumerate(domains):
            if i%100 < int(self.prop * 40 / 100):
                train_domains.append(domain)
            elif i%100 >= int(90 * 40 / 100):
                val_domains.append(domain)
            #else:
            #    test_domains.append(domain)
        def get_data_dict(domains):
            img, msk, mtd = flair_gather_data(
                domains, path_metadata=None, use_metadata=False, test_set=False
            )
            return {"IMG":img, "MSK":msk}
        self.dict_train = get_data_dict(train_domains)
        self.dict_val = get_data_dict(val_domains)
        #self.dict_test = get_data_dict(test_domains)

    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_set = datasets.Flair(
                self.dict_train["IMG"],
                self.dict_train["MSK"],
                self.bands,
                self.merge,
                transforms=self.train_tf,
            )

            self.val_set = datasets.Flair(
                self.dict_val["IMG"],
                self.dict_val["MSK"],
                self.bands,
                self.merge,
                transforms=self.val_tf,
            )
        if stage in ("test", "predict"):
            dataset = DatasetFlair2(
                self.dict_test["IMG"],
                self.dict_test["MSK"],
                self.bands,
                self.merge,
                transforms=self.test_tf,
            )
            if stage == "test":
                self.test_set = dataset
            else:
                self.pred_set = dataset
                
    def get_loader(self, dataset):
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
        train_dataloaders["sup"] = self.get_loader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        return CombinedLoader(train_dataloaders, mode="min_size")
    
    def val_dataloader(self):
        return self.get_loader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return self.get_loader(self.pred_set)(
            shuffle=False,
            drop_last=False,
        )
    
    def test_dataloader(self):
        return self.get_loader(self.test_set)(
            shuffle=False,
            drop_last=False,
        )

class DatamoduleFlair2(DatamoduleFlair1):
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        #assert 0 < self.prop < 90
        
    def prepare_data(self):
        domains = [self.data_path / "FLAIR_1" / "train" / d for d in self.train_domains]
        all_img, all_msk, all_mtd = flair_gather_data(
            domains, path_metadata=None, use_metadata=False, test_set=False
        )
        self.dict_train = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_val = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_test = {"IMG": [], "MSK": [], "MTD": []}
        for i, (img, msk) in enumerate(zip(all_img, all_msk)):
            if i%100 < self.prop:
                self.dict_train["IMG"].append(img)
                self.dict_train["MSK"].append(msk)
            elif i%100 >= 90:
                self.dict_val["IMG"].append(img)
                self.dict_val["MSK"].append(msk)
            else:
                self.dict_test["IMG"].append(img)
                self.dict_test["MSK"].append(msk)

class DatamoduleFlair2Semisup(DatamoduleFlair2):
    
    def __init__(
        self,
        unlabeled_prop,
        pseudo_label_dir=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.unlabeled_prop = unlabeled_prop
        
    def prepare_data(self):
        domains = [self.data_path / "FLAIR_1" / "train" / d for d in self.train_domains]
        all_img, all_msk, all_mtd = flair_gather_data(
            domains, path_metadata=None, use_metadata=False, test_set=False
        )
        self.dict_train = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_train_unlabeled = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_val = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_test = {"IMG": [], "MSK": [], "MTD": []}
        for i, (img, msk) in enumerate(zip(all_img, all_msk)):
            if self.prop <= i%100 <= self.prop + self.unlabeled_prop:
                self.dict_train_unlabeled["IMG"].append(img)
            if i%100 < self.prop:
                self.dict_train["IMG"].append(img)
                self.dict_train["MSK"].append(msk)
            elif i%100 >= 90:
                self.dict_val["IMG"].append(img)
                self.dict_val["MSK"].append(msk)
            else:
                self.dict_test["IMG"].append(img)
                self.dict_test["MSK"].append(msk)

    def setup(self, stage):
        super().setup(stage)
        if stage in ("fit"):
            self.unlabeled_set = datasets.Flair(
                self.dict_train_unlabeled["IMG"],
                [],
                self.bands,
                self.merge,
                #self.crop_size,
                transforms=self.train_tf,
            )
        
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.get_loader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        train_dataloaders["unsup"] = self.get_loader(self.unlabeled_set)(
            shuffle=True,
            drop_last=True
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
