import json

import os
import random
import re
from os.path import join
from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate


#### encode metadata
def coordenc_opt(coords, enc_size=32) -> np.array:
    d = int(enc_size / 2)
    d_i = np.arange(0, d / 2)
    freq = 1 / (10e7 ** (2 * d_i / d))

    x, y = coords[0] / 10e7, coords[1] / 10e7
    enc = np.zeros(d * 2)
    enc[0:d:2] = np.sin(x * freq)
    enc[1:d:2] = np.cos(x * freq)
    enc[d::2] = np.sin(y * freq)
    enc[d + 1 :: 2] = np.cos(y * freq)
    return list(enc)

def norm_alti(alti: int) -> float:
    min_alti = 0
    max_alti = 3164.9099121094
    return [(alti - min_alti) / (max_alti - min_alti)]

def format_cam(cam: str) -> np.array:
    return [[1, 0] if "UCE" in cam else [0, 1]][0]

def cyclical_enc_datetime(date: str, time: str) -> list:
    def norm(num: float) -> float:
        return (num - (-1)) / (1 - (-1))

    year, month, day = date.split("-")
    if year == "2018":
        enc_y = [1, 0, 0, 0]
    elif year == "2019":
        enc_y = [0, 1, 0, 0]
    elif year == "2020":
        enc_y = [0, 0, 1, 0]
    elif year == "2021":
        enc_y = [0, 0, 0, 1]
    sin_month = np.sin(2 * np.pi * (int(month) - 1 / 12))  ## months of year
    cos_month = np.cos(2 * np.pi * (int(month) - 1 / 12))
    sin_day = np.sin(2 * np.pi * (int(day) / 31))  ## max days
    cos_day = np.cos(2 * np.pi * (int(day) / 31))
    h, m = time.split("h")
    sec_day = int(h) * 3600 + int(m) * 60
    sin_time = np.sin(2 * np.pi * (sec_day / 86400))  ## total sec in day
    cos_time = np.cos(2 * np.pi * (sec_day / 86400))
    return enc_y + [
        norm(sin_month),
        norm(cos_month),
        norm(sin_day),
        norm(cos_day),
        norm(sin_time),
        norm(cos_time),
    ]

def _gather_data(
    path_folders, path_metadata: str, use_metadata: bool, test_set: bool
) -> dict:
    #### return data paths
    def get_data_paths(path, filter):
        for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

    img, msk, mtd = [], [], []
    if path_folders:
        for domain in path_folders:
            # list_img = sorted(list(get_data_paths(domain, 'IMG*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
            list_img = sorted(
                list(get_data_paths(domain, "IMG*.tif")),
                key=lambda x: int(x.split("_")[-1][:-4]),
            )
            img += list_img
            if test_set == False:
                # list_msk = sorted(list(get_data_paths(domain, 'MSK*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
                list_msk = sorted(
                    list(get_data_paths(domain, "MSK*.tif")),
                    key=lambda x: int(x.split("_")[-1][:-4]),
                )
                msk += list_msk
            # print(f'domain {domain}: {[(img, msk) for img, msk in zip(list_img, list_msk)]}')
            # break

        if use_metadata == True:
            with open(path_metadata, "r") as f:
                metadata_dict = json.load(f)
            for img in data["IMG"]:
                curr_img = img.split("/")[-1][:-4]
                enc_coords = coordenc_opt(
                    [
                        metadata_dict[curr_img]["patch_centroid_x"],
                        metadata_dict[curr_img]["patch_centroid_y"],
                    ]
                )
                enc_alti = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
                enc_camera = format_cam(metadata_dict[curr_img]["camera"])
                enc_temporal = cyclical_enc_datetime(
                    metadata_dict[curr_img]["date"], metadata_dict[curr_img]["time"]
                )
                mtd_enc = enc_coords + enc_alti + enc_camera + enc_temporal
                mtd.append(mtd_enc)

    return img, msk, mtd


class DatamoduleFlair1(LightningDataModule):
    train_domains = [
        "D004_2021",
        "D014_2020",
        "D029_2021",
        "D031_2019",
        "D058_2020",
        "D077_2021",
        "D067_2021",
        "D066_2021",
        "D033_2021",
        "D055_2018",
        "D072_2019",
        "D044_2020",
        "D017_2018",
        "D086_2020",
        "D049_2020",
        "D016_2020",
        "D063_2019",
        "D091_2021",
        "D070_2020",
        "D013_2020",
        "D023_2020",
        "D074_2020",
        "D021_2020",
        "D080_2021",
        "D078_2021",
        "D032_2019",
        "D081_2020",
        "D046_2019",
        "D052_2019",
        "D051_2019",
        "D038_2021",
        "D009_2019",
        "D034_2021",
        "D006_2020",
        "D008_2019",
        "D041_2021",
        "D035_2020",
        "D007_2020",
        "D060_2021",
        "D030_2021",
    ]
    #test_domains = [
    #    "D012_2019",
    #    "D022_2021",
    #    "D026_2020",
    #    "D064_2021",
    #    "D068_2021",
    #    "D071_2020",
    #    "D075_2021",
    #    "D076_2019",
    #    "D083_2020",
    #    "D085_2019"
    #]

    def __init__(
        self,
        data_path,
        merge,
        prop,
        bands,
        #crop_size,
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
        self.classes = DatasetFlair2.classes[merge].value
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
            else:
                test_domains.append(domain)
        def get_data_dict(domains):
            img, msk, mtd = _gather_data(
                domains, path_metadata=None, use_metadata=False, test_set=False
            )
            return {"IMG":img, "MSK":msk}
        self.dict_train = get_data_dict(train_domains)
        self.dict_val = get_data_dict(val_domains)
        self.dict_test = get_data_dict(test_domains)

    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_set = DatasetFlair2(
                self.dict_train["IMG"],
                self.dict_train["MSK"],
                self.bands,
                self.merge,
                transforms=self.train_tf,
            )

            self.val_set = DatasetFlair2(
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
        all_img, all_msk, all_mtd = _gather_data(
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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.unlabeled_prop = unlabeled_prop
        
    def prepare_data(self):
        domains = [self.data_path / "FLAIR_1" / "train" / d for d in self.train_domains]
        all_img, all_msk, all_mtd = _gather_data(
            domains, path_metadata=None, use_metadata=False, test_set=False
        )
        self.dict_train = {"IMG": [], "MSK": [], "MTD": []}
        self.dict_train_unlabeled = {"IMG": [], "MTD": []}
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
            self.unlabeled_set = DatasetFlair2(
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