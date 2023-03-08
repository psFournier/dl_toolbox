import torch
import rasterio
import numpy as np

from rasterio.windows import Window
from argparse import ArgumentParser 

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.augmentations as augmentations
    
class RasterDs(torch.utils.data.Dataset):
    """
    A class to represent 

    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(
        self,
        image_path,
        tile,
        crop_size,
        mins,
        maxs,
        img_aug=None,
        label_path=None,
        fixed_crops=False,
        crop_step=None
    ):

        self.image_path = image_path
        self.tile = tile
        self.mins = np.array(mins).reshape(-1, 1, 1)
        self.maxs = np.array(maxs).reshape(-1, 1, 1)
        self.crop_size = crop_size
        self.img_aug = augmentations.get_transforms(img_aug)
        self.label_path = label_path
        
        with rasterio.open(image_path) as raster: 
            self.raster_tf = raster.transform
            
        if fixed_crops:
            self.fixed_crops = list(
                get_tiles(
                    nols=tile.width, 
                    nrows=tile.height, 
                    size=crop_size, 
                    step=crop_step if crop_step else crop_size,
                    row_offset=tile.row_off, 
                    col_offset=tile.col_off
                )
            )
        else:
            self.fixed_crops=None

    def __len__(self):

        return len(self.fixed_crops) if self.fixed_crops else 1 # Attention 1 ou la taille du dataset pour le concat

    def __getitem__(self, idx):
        
        if self.fixed_crops:
            crop = self.fixed_crops[idx]
        else:
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            crop = Window(cx, cy, self.crop_size, self.crop_size)
        
        crop_tf = rasterio.windows.transform(crop, transform=self.raster_tf)
            
        image = self.read_image(self.image_path, crop)
        image = torch.from_numpy(image).float().contiguous()

        label = None
        if self.label_path:
            label = self.read_label(self.label_path, crop)
            label = torch.from_numpy(label).long().contiguous()

        if self.img_aug is not None:
            image, label = self.img_aug(img=image, label=label)

        return {
            'image':image,
            'label':label,
            'crop':crop,
            'crop_tf':crop_tf,
            'path': self.image_path
        }