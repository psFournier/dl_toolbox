import torch
import rasterio
import imagesize
import numpy as np

from rasterio.windows import Window
from argparse import ArgumentParser 

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
from dl_toolbox.torch_datasets.utils import *

    
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
        crop_step=None,
        #one_hot=False,
        #*args,
        #**kwargs
    ):

        self.image_path = image_path
        self.tile = tile
        self.mins = np.array(mins).reshape(-1, 1, 1)
        self.maxs = np.array(maxs).reshape(-1, 1, 1)
        self.crop_size = crop_size
        self.img_aug = get_transforms(img_aug)
        self.label_path = label_path
        self.crop_windows = list(get_tiles(
            nols=tile.width, 
            nrows=tile.height, 
            size=crop_size, 
            step=crop_step if crop_step else crop_size,
            row_offset=tile.row_off, 
            col_offset=tile.col_off)) if fixed_crops else None
        #self.labels_to_rgb = LabelsToRGB(self.labels)
        #self.rgb_to_labels = RGBToLabels(self.labels)      

    def __len__(self):

        return len(self.crop_windows) if self.crop_windows else 1 # Attention 1 ou la taille du dataset pour le concat

    def __getitem__(self, idx):
        
        if self.crop_windows:
            window = self.crop_windows[idx]
        else:
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
            
        image = self.read_image(self.image_path, window)
        image = torch.from_numpy(image).float().contiguous()

        label = None
        if self.label_path:
            label = self.read_label(self.label_path, window)
            label = torch.from_numpy(label).long().contiguous()

        if self.img_aug is not None:
            end_image, end_mask = self.img_aug(img=image, label=label)
        else:
            end_image, end_mask = image, label

        return {
            'orig_image':image,
            'orig_mask':label,
            'image':end_image,
            'window':window,
            'mask':end_mask,
            'path': self.image_path
        }