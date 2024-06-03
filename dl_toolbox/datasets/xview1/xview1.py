from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from pathlib import Path
import torch
import torchvision.transforms.v2 as v2
import rasterio
import numpy as np

from dl_toolbox.utils import label, merge_labels




LABEL_TO_STRING = {
    11: "Fixed-wing Aircraft",
    12: "Small Aircraft",
    13: "Passenger/Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle",
    18: "Small Car",
    19: "Bus",
    20: "Pickup Truck",
    21: "Utility Truck",
    23: "Truck",
    24: "Cargo Truck",
    25: "Truck Tractor w/ Box Trailer",
    26: "Truck Tractor",
    27: "Trailer",
    28: "Truck Tractor w/ Flatbed Trailer",
    29: "Truck Tractor w/ Liquid Tank",
    32: "Crane Truck",
    33: "Railway Vehicle",
    34: "Passenger Car",
    35: "Cargo/Container Car",
    36: "Flat Car",
    37: "Tank car",
    38: "Locomotive",
    40: "Maritime Vessel",
    41: "Motorboat",
    42: "Sailboat",
    44: "Tugboat",
    45: "Barge",
    47: "Fishing Vessel",
    49: "Ferry",
    50: "Yacht",
    51: "Container Ship",
    52: "Oil Tanker",
    53: "Engineering Vehicle",
    54: "Tower crane",
    55: "Container Crane",
    56: "Reach Stacker",
    57: "Straddle Carrier",
    59: "Mobile Crane",
    60: "Dump Truck",
    61: "Haul Truck",
    62: "Scraper/Tractor",
    63: "Front loader/Bulldozer",
    64: "Excavator",
    65: "Cement Mixer",
    66: "Ground Grader",
    71: "Hut/Tent",
    72: "Shed",
    73: "Building",
    74: "Aircraft Hangar",
    76: "Damaged Building",
    77: "Facility",
    79: "Construction Site",
    83: "Vehicle Lot",
    84: "Helipad",
    86: "Storage Tank",
    89: "Shipping container lot",
    91: "Shipping Container",
    93: "Pylon",
    94: "Tower",
}

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = defaultdict(list)
    for dct in list_of_dicts:
        for key, value in dct.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)

class xView1(Dataset):
    """
    Requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
    """
    classes = [label(v, (0, int(255-k), 255), {k}) for k, v in LABEL_TO_STRING.items()]

    def __init__(self, root, annFile, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.merges = [list(l.values) for l in self.classes]
        self.transforms = v2.ToDtype(
            dtype={tv_tensors.Image: torch.float32, "others":None},
            scale=True
        )
        if transforms:
            self.transforms = v2.Compose([self.transforms, transforms])

    def __getitem__(self, index):
        id = self.ids[index]
        path = self.root/self.coco.loadImgs(id)[0]["file_name"]
        with rasterio.open(path, "r") as file:
            image = file.read(out_dtype=np.uint8)
        tv_image = tv_tensors.Image(torch.from_numpy(image))
                
        target = self.coco.loadAnns(self.coco.getAnnIds(id))
        target = list_of_dicts_to_dict_of_lists(target)
        tv_target = {}
        tv_target["boxes"] = tv_tensors.BoundingBoxes(
            target["bbox"],
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=tuple(F.get_size(tv_image)),
        )
        labels = torch.tensor(target["category_id"])
        tv_target['labels'] = merge_labels(labels, self.merges).long()
        if self.transforms is not None:
            tv_image, tv_target = self.transforms(tv_image, tv_target)
        return tv_image, tv_target, path

    def __len__(self):
        return len(self.ids)