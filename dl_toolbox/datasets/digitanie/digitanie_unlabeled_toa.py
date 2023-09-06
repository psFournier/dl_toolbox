import numpy as np
import rasterio
import torch
import enum
import random
from torch.utils.data import Dataset

from dl_toolbox.utils import label, get_tiles, merge_labels
import rasterio.windows as W
from fiona.transform import transform as f_transform
import rasterio.warp
from shapely.geometry import box
from rasterio.enums import Resampling

from .digitanie import Digitanie
    
class DigitanieUnlabeledToa(Digitanie):
    
    def __init__(
        self,
        toa,
        bands,
        windows,
        transforms
    ):
        self.toa = toa
        self.bands = bands
        self.transforms = transforms
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        with rasterio.open(self.toa, "r") as file:
            image = file.read(window=window, out_dtype=np.int16, indexes=self.bands)
        image = torch.from_numpy(image).float()
        image, label = self.transforms(img=image, label=None)
        return {"image": image, "label": None}
    
class DigitanieSublabeledToaDataset(Dataset):
    classes = classes
    
    def __init__(
        self,
        toa,
        msks,
        bands,
        merge,
        transforms
    ):
        self.toa = toa
        self.msks = msks
        self.bands = bands
        self.transforms = transforms
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.crop_size = 512

    def __len__(self):
        return len(self.msks)

    def __getitem__(self, idx):   
        msk = rasterio.open(self.msks[idx])
        toa = rasterio.open(self.toa)
        
        rows, cols = msk.shape
        crop_size = self.crop_size * toa.res[0] / msk.res[0]
        cx = random.randint(0, int(rows - crop_size))
        cy = random.randint(0, int(cols - crop_size))
        window = W.Window(cx, cy, crop_size, crop_size)
        
        label = msk.read(
            window=window,
            out_shape=(
                msk.count,
                self.crop_size,
                self.crop_size
            ),
            resampling=Resampling.nearest,
            out_dtype=np.uint8
        )
        
        minx, miny, maxx, maxy = W.bounds(window, transform=msk.transform)
        if msk.crs != toa.crs:
            (minx, maxx), (miny, maxy) = f_transform(
                msk.crs, toa.crs, [minx, maxx], [miny, maxy]
            )
        window = W.from_bounds(minx, miny, maxx, maxy, transform=toa.transform)
        image = toa.read(window=window, out_dtype=np.int16, indexes=self.bands)
        
        image = torch.clip(torch.from_numpy(image).float() / 8000., 0, 1)
        label = merge_labels(label, self.merges)
        label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        msk.close()
        toa.close()
        return {
            "image": image,
            "label": label.squeeze()
        }
            
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
