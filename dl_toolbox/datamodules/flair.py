import json

import os
import random
import re
from os.path import join
from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate


def _gather_data(
    path_folders, path_metadata: str, use_metadata: bool, test_set: bool
) -> dict:
    #### return data paths
    def get_data_paths(path, filter):
        for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

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

    data = {"IMG": [], "MSK": [], "MTD": []}
    if path_folders:
        for domain in path_folders:
            # list_img = sorted(list(get_data_paths(domain, 'IMG*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
            list_img = sorted(
                list(get_data_paths(domain, "IMG*.tif")),
                key=lambda x: int(x.split("_")[-1][:-4]),
            )
            data["IMG"] += list_img
            if test_set == False:
                # list_msk = sorted(list(get_data_paths(domain, 'MSK*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
                list_msk = sorted(
                    list(get_data_paths(domain, "MSK*.tif")),
                    key=lambda x: int(x.split("_")[-1][:-4]),
                )
                data["MSK"] += list_msk
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
                data["MTD"].append(mtd_enc)

        if test_set == False:
            if len(data["IMG"]) != len(data["MSK"]):
                print(
                    "[WARNING !!] UNMATCHING NUMBER OF IMAGES AND MASKS ! Please check load_data function for debugging."
                )
            if (
                data["IMG"][0][-10:-4] != data["MSK"][0][-10:-4]
                or data["IMG"][-1][-10:-4] != data["MSK"][-1][-10:-4]
            ):
                print(
                    "[WARNING !!] UNSORTED IMAGES AND MASKS FOUND ! Please check load_data function for debugging."
                )

    return data


class Flair(LightningDataModule):
    baseline_val_domains = [
        "D004_2021",
        "D014_2020",
        "D029_2021",
        "D031_2019",
        "D058_2020",
        "D077_2021",
        "D067_2021",
        "D066_2021",
    ]
    baseline_train_domains = [
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

    def __init__(
        self,
        data_path,
        merge,
        bands,
        crop_size,
        train_tf,
        val_tf,
        batch_size,
        num_workers,
        pin_memory,
        class_weights=None
        # train_domains,
        # val_domains,
        # test_domains,
        # unsup_train_idxs=None,
        # *args,
        # **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers
        self.bands = bands

        domains = [
            data_path / "train" / domain
            for domain in self.baseline_val_domains + self.baseline_train_domains
        ]
        # shuffle(domains)
        idx_split = int(len(domains) * 0.95)
        train_domains, val_domains = domains[:idx_split], domains[idx_split:]

        dict_train = _gather_data(
            train_domains, path_metadata=None, use_metadata=False, test_set=False
        )
        dict_val = _gather_data(
            val_domains, path_metadata=None, use_metadata=False, test_set=False
        )

        self.train_set = datasets.Flair2(
            dict_train["IMG"],
            dict_train["MSK"],
            bands,
            merge,
            crop_size,
            shuffle=False,
            transforms=train_tf,
        )
        # self.train_set = ConcatDataset([
        #    datasets.Flair(
        #        img,
        #        msk,
        #        bands,
        #        merge,
        #        crop_size,
        #        shuffle=False,
        #        transforms=train_tf,
        #    ) for img, msk in zip(dict_train['IMG'], dict_train['MSK'])
        # ])

        self.val_set = datasets.Flair2(
            dict_val["IMG"],
            dict_val["MSK"],
            bands,
            merge,
            crop_size,
            shuffle=False,
            transforms=val_tf,
        )
        # self.val_set = ConcatDataset([
        #    datasets.Flair(
        #        img,
        #        msk,
        #        bands,
        #        merge,
        #        crop_size,
        #        shuffle=False,
        #        transforms=val_tf,
        #    ) for img, msk in zip(dict_val['IMG'], dict_val['MSK'])
        # ])

        self.in_channels = len(self.bands)
        self.classes = datasets.Flair.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )

    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=CustomCollate(),
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return CombinedLoader(train_dataloaders, mode="min_size")

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    # def predict_dataloader(self):
    #
    #    predict_dataloader = DataLoader(
    #        dataset=self.test_set,
    #        shuffle=False,
    #        batch_size=8,
    #        collate_fn=CustomCollate(),
    #        num_workers=self.num_workers
    #    )
    #
    #    return predict_dataloader


class OCS_DataModule(LightningDataModule):
    def __init__(
        self,
        dict_train=None,
        dict_val=None,
        dict_test=None,
        num_workers=1,
        batch_size=2,
        drop_last=True,
        num_classes=13,
        num_channels=5,
        use_metadata=True,
        use_augmentations=True,
    ):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes, self.num_channels = num_classes, num_channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_metadata = use_metadata
        self.use_augmentations = use_augmentations

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = Fit_Dataset(
                dict_files=self.dict_train,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
                use_augmentations=self.use_augmentations,
            )

            self.val_dataset = Fit_Dataset(
                dict_files=self.dict_val,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
            )

        elif stage == "predict":
            self.pred_dataset = Predict_Dataset(
                dict_files=self.dict_test,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
