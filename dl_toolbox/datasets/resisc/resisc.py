import enum
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import DatasetFolder

import dl_toolbox.transforms as transforms

from dl_toolbox.utils import label


cls_names = ["airplane", "bridge", "commercial_area", "golf_course", "island", "mountain", "railway_station", "sea_ice", "storage_tank", "airport", "chaparral", "dense_residential", "ground_track_field", "lake", "overpass", "rectangular_farmland", "ship", "tennis_court", "baseball_diamond", "church", "desert", "harbor", "meadow", "palace", "river", "snowberg", "terrace", "basketball_court", "circular_farmland", "forest", "industrial_area", "medium_residential", "parking_lot", "roundabout", "sparse_residential", "thermal_power_station", "beach", "cloud", "freeway", "intersection", "mobile_home_park", "railway", "runway", "stadium", "wetland"]

all45 = [label(name, None, {i}) for i, name in enumerate(cls_names)]
test = [
    label("air", None, {0, 9}),
    label("water", None, {44})
]

classes = enum.Enum(
    "Resisc",
    {
        "all45": all45,
        "test": test
    },
)

def pil_to_torch_loader(path: str):
    with open(path, "rb") as f:
        img = np.array(Image.open(f))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img


class Resisc(DatasetFolder):
    
    classes = classes
    
    def __init__(self, data_path, transforms, merge):
        self.class_list = self.classes[merge].value
        super().__init__(
            root=data_path, loader=pil_to_torch_loader, extensions=("jpg",)
        )
        self.transforms = transforms

    def find_classes(self, directory):
        names = [cls_names[i] for label in self.class_list for i in label.values]
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and entry.name in names
        )
        class_to_idx = {
            cls_names[i]: j
            for j, label in enumerate(self.class_list)
            for i in label.values
            if cls_names[i] in classes
        }
        return classes, class_to_idx

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        #image = self.loader(path)/255.
        image = self.loader(path)
        #image = self.transforms(image)
        image,_ = self.transforms(image)
        return {
            "image": image,
            "label": torch.tensor(label),
            "path": path
        }