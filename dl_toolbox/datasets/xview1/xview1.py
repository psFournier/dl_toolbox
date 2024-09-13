from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from pathlib import Path
import torch
import torchvision.transforms.v2 as v2
import rasterio
import numpy as np
import enum
from functools import reduce

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

all60 = [label(v, (0, int(255-k), 255), {k}) for k, v in LABEL_TO_STRING.items()]

building = [label("building", (0, 255, 0), {73})]


classes = enum.Enum(
    "xViewClasses",
    {
        "all": all60,
        "building": building,
    },
)

class xView1(Dataset):
    """
    Requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
    """
    classes = classes

    def __init__(self, root, annFile, transforms=None, merge='all'):
        self.root = Path(root)
        self.transforms = transforms
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = v2.ToDtype(
            dtype={tv_tensors.Image: torch.float32, "others":None},
            scale=True
        )
        if transforms:
            self.transforms = v2.Compose([self.transforms, transforms])
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        
    def merge(self, labels, boxes):
        """
        Args: 
            labels: tensor shape L
            boxes: tensor shape Lx4
        Returns:
            
        """
        merged_labels = []
        merged_boxes = []
        for i, l in enumerate(self.class_list): 
            # indices of tgts whose label belongs to the i-th merge
            idx = reduce(torch.logical_or, [labels == v for v in l.values])
            # i+1 because in detection, class label 0 should be left for no-obj in algos
            merged_labels.append((i+1) * torch.ones_like(labels[idx]))
            merged_boxes.append(boxes[idx])
        merged_labels = torch.cat(merged_labels, dim=0)
        merged_boxes = torch.cat(merged_boxes, dim=0)
        return merged_labels, merged_boxes

    def __getitem__(self, index):
        id = self.ids[index]
        path = self.root/self.coco.loadImgs(id)[0]["file_name"]
        with rasterio.open(path, "r") as file:
            image = file.read(out_dtype=np.uint8)
        tv_image = tv_tensors.Image(torch.from_numpy(image))
                
        target = self.coco.loadAnns(self.coco.getAnnIds(id))
        target = list_of_dicts_to_dict_of_lists(target)
        labels = torch.tensor(target["category_id"])
        boxes = torch.as_tensor(target["bbox"]).float()
        merged_labels, merged_boxes = self.merge(labels, boxes)
        tv_target = {}
        tv_target["boxes"] = tv_tensors.BoundingBoxes(
            merged_boxes,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=tuple(F.get_size(tv_image)),
        )
        tv_target['labels'] = merged_labels.long()
        if self.transforms is not None:
            tv_image, tv_target = self.transforms(tv_image, tv_target)
        return tv_image, tv_target, path

    def __len__(self):
        return len(self.ids)