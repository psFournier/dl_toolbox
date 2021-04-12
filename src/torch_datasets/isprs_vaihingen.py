import os
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
from torch_datasets import BaseDataset, BaseDatasetLabeled, BaseDatasetUnlabeled




class IsprsVaihingen(BaseDataset, ABC):

    labeled_image_paths = [
        'top/top_mosaic_09cm_area{}.tif'.format(i) for i in [
            1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
            34, 37
        ]
    ]
    unlabeled_image_paths = [
        'top/top_mosaic_09cm_area{}.tif'.format(i) for i in [
            2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29,
            31, 33, 35, 38
        ]
    ]
    label_paths = [
        'gts_for_participants/top_mosaic_09cm_area{}.tif'.format(i) for i in [
            1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
            34, 37
        ]
    ]
    image_size = (1900,2600)

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs
        )
        # self.mean_labeled_pixels = [0.4727, 0.3205, 0.3159]
        # self.std_labeled_pixels = [0.2100, 0.1496, 0.1426]

    @classmethod
    def colors_to_labels(cls, labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        colors = [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [255, 255, 0],
            [0, 255, 0],
            [255, 0, 0],
        ]

        for id_col, col in enumerate(colors):
            d = labels_color[:, :, 0] == col[0]
            d = np.logical_and(d, (labels_color[:, :, 1] == col[1]))
            d = np.logical_and(d, (labels_color[:, :, 2] == col[2]))
            labels[d] = id_col

        return labels


class IsprsVaihingenUnlabeled(IsprsVaihingen, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class IsprsVaihingenLabeled(IsprsVaihingen, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)