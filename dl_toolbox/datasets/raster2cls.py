import random

import torch
import rasterio
import numpy as np
import shapely
from scipy import stats

import rasterio.windows as windows
from argparse import ArgumentParser
import torchvision.transforms.functional as F

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.augmentations as augmentations
from dl_toolbox.utils import minmax
from dl_toolbox.datasets import Raster


class Raster2Cls(Raster):
    def __init__(self, focus_rad, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focus_rad = focus_rad

    def get_focus(self, r, cx, cy):
        xx, yy = np.mgrid[: self.crop_size, : self.crop_size]

        circle = (xx - cx) ** 2 + (yy - cy) ** 2
        focus = torch.zeros((self.crop_size, self.crop_size))
        focus[circle < r**2] = 1

        # focus = stats.multivariate_normal.pdf(
        #    np.dstack((xx, yy)),
        #    mean=np.array([cx, cy]),
        #    cov=r**2
        # )
        # focus = torch.from_numpy(focus).float()

        return focus

    def __getitem__(self, idx):
        pre_crop_size = int(np.ceil(np.sqrt(2) * self.crop_size))
        crop = self.get_crop(pre_crop_size)
        pre_image, pre_label = self.read_crop(crop)
        image, label = self.rnd_rotate_and_crop(pre_image, self.crop_size, pre_label)
        image = self.normalize(image)

        if self.aug is not None:
            image, label = self.aug(img=image, label=label)

        # cx, cy = np.random.randint(self.crop_size, size=2)
        cx, cy = self.crop_size // 2, self.crop_size // 2
        focus = self.get_focus(self.focus_rad, cx, cy)
        cols, rows = np.meshgrid(np.arange(self.crop_size), np.arange(self.crop_size))
        image = torch.vstack(
            [
                image,
                # focus.unsqueeze(dim=0),
                # torch.from_numpy(cols).float().unsqueeze(dim=0),
                # torch.from_numpy(rows).float().unsqueeze(dim=0),
            ]
        )

        if label is not None:
            # c=self.crop_size
            pre_label = label.squeeze().long()
            # center_crop = label[c//2:c+c//2, c//2:c+c//2]
            bincounts = torch.bincount(
                pre_label.flatten(),
                minlength=len(self.nomenclature),
                weights=focus.flatten(),
            )
            bincounts /= sum(bincounts)
            # label = torch.argmax(bincounts)
            label = torch.Tensor(bincounts[1] > 0.2).long()
            # print(bincounts, label)

        crop_transform = windows.transform(
            crop, transform=self.data_src.meta["transform"]
        )

        return {
            "pre_image": pre_image,
            "pre_label": pre_label,
            "image": image,
            "label": label,
            "crop": crop,
            "crop_tf": crop_transform,
            "path": self.data_src.image_path,
            "crs": self.data_src.meta["crs"],
        }


class PretiledRaster2Cls(Raster2Cls):
    def __init__(self, crop_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_step = crop_step
        zone = self.data_src.zone
        assert isinstance(zone, windows.Window)

        self.tiles = [
            tile
            for tile in get_tiles(
                nols=zone.width,
                nrows=zone.height,
                size=self.crop_size,
                step=crop_step if crop_step else self.crop_size,
                row_offset=zone.row_off,
                col_offset=zone.col_off,
            )
        ]

        ranges = range(self.focus_rad, self.crop_size - self.focus_rad, self.focus_rad)
        self.focuses = [
            self.get_focus(self.focus_rad, cx, cy) for cx in ranges for cy in ranges
        ]

    def __len__(self):
        return len(self.tiles) * len(self.focuses)

    def __getitem__(self, idx):
        crop = self.tiles[idx]

        image, label = self.read_crop(crop)
        image = self.normalize(image)

        if self.aug is not None:
            image, label = self.aug(img=image, label=label)
        if label is not None:
            label = label.long().squeeze()

        crop_transform = windows.transform(
            crop, transform=self.data_src.meta["transform"]
        )

        return {
            "image": image,
            "label": label,
            "crop": crop,
            "crop_tf": crop_transform,
            "path": self.data_src.image_path,
            "crs": self.data_src.meta["crs"],
        }
