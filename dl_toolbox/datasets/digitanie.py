import numpy as np
import rasterio
import torch
import enum
from torch.utils.data import Dataset

from dl_toolbox.utils import label, get_tiles, merge_labels
from rasterio.windows import Window

all9 = [
    label("nodata", (250, 250, 250), {0}),
    label("bare_ground", (100, 50, 0), {1}),
    label("low_vegetation", (0, 250, 50), {2}),
    label("water", (0, 50, 250), {3}),
    label("building", (250, 50, 50), {4}),
    label("high_vegetation", (0, 100, 50), {5}),
    label("parking", (200, 200, 200), {6}),
    label("road", (100, 100, 100), {7}),
    label("railways", (200, 100, 200), {8}),
    label("swimmingpool", (50, 150, 250), {9}),
]

main6 = [
    label("other", (0, 0, 0), {0, 1, 6, 9}),
    label("low vegetation", (0, 250, 50), {2}),
    label("high vegetation", (0, 100, 50), {5}),
    label("water", (0, 50, 250), {3}),
    label("building", (250, 50, 50), {4}),
    label("road", (100, 100, 100), {7}),
    label("railways", (200, 100, 200), {8}),
]


classes = enum.Enum(
    "Digitanie9Classes",
    {
        "all9": all9,
        "main6": main6
    }
)

class DigitanieDataset(Dataset):
    classes = classes

    def __init__(
        self,
        imgs,
        msks,
        bands,
        merge,
        transforms,
    ):
        self.imgs = imgs
        self.msks = msks
        self.bands = bands
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        with rasterio.open(self.imgs[idx], "r") as file:
            image = file.read(
                out_dtype=np.int16,
                indexes=self.bands
            )
        image = torch.clip(torch.from_numpy(image).float() / 8000., 0, 1)
        label = None
        if self.msks:
            with rasterio.open(self.msks[idx], "r") as file:
                label = file.read(out_dtype=np.uint8)
            label = merge_labels(label, self.merges)
            label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze()
        }
    
class DigitanieToaDataset(Dataset):
    classes = classes
    
    def __init__(
        self,
        img,
        bands,
        windows,
        transforms
    ):
        self.img = img
        self.bands = bands
        self.transforms = transforms
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        with rasterio.open(self.img, "r") as file:
            image = file.read(window=window, out_dtype=np.int16, indexes=self.bands)
        image = torch.clip(torch.from_numpy(image).float() / 8000., 0, 1)
        image, label = self.transforms(img=image, label=None)
        return {"image": image, "label": None}
    
    
#class Digitanie(Dataset):
#    def __init__(
#        self, data_src, bands, merge, crop_size, shuffle, transforms, crop_step=None
#    ):
#        self.data_src = data_src
#        self.bands = bands
#        self.merge = merge
#        self.crop_size = crop_size
#        self.shuffle = shuffle
#        self.transforms = transforms
#
#        h = data_src.zone.height
#        w = data_src.zone.width
#        self.nb_crops = h * w / (crop_size**2)
#
#        if shuffle:
#            self.get_crop = RandomCropFromWindow(data_src.zone, crop_size)
#        else:
#            self.get_crop = FixedCropFromWindow(data_src.zone, crop_size, crop_step)
#
#    def read_crop(self, crop):
#        image = self.data_src.read_image(crop, bands=self.bands)
#        image = torch.from_numpy(image)
#
#        label = None
#        if self.data_src.label_path:
#            label = self.data_src.read_label(crop, merge=self.merge)
#            label = torch.from_numpy(label)
#
#        return image, label
#
#    def __len__(self):
#        if self.shuffle:
#            return int(self.nb_crops)
#        else:
#            return len(self.get_crop.crops)
#
#    def __getitem__(self, idx):
#        crop = self.get_crop(idx)
#        raw_image, raw_label = self.read_crop(crop)
#        image, label = self.transforms(img=raw_image, label=raw_label)
#
#        # Here transform means the affine transform from matrix coords to long/lat
#        crop_transform = rasterio.windows.transform(
#            crop, transform=self.data_src.meta["transform"]
#        )
#
#        return {
#            "image": image,
#            "label": label,
#            "crop": crop,
#            "crop_tf": crop_transform,
#            "path": self.data_src.image_path,
#            "crs": self.data_src.meta["crs"],
#        }
