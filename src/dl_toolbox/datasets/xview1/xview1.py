from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from pathlib import Path
import torch
import rasterio
import numpy as np
import enum

from dl_toolbox.utils import label, merge_labels_boxes
from functools import reduce
import rasterio.windows as win



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



class xView1(torch.utils.data.Dataset):
    """
    Requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
    """
    classes = classes
    
    def __init__(self, root, coco_dataset, ids, transforms=None, merge='all'):
        self.root = Path(root)
        self.coco = coco_dataset
        self.ids_and_windows = ids
        self.transforms = transforms
        self.class_list = self.classes[merge].value
        self.init_tf(transforms)
        
    def init_tf(self, tf):
        self.transforms = T.ToDtype(
            dtype={tv_tensors.Image: torch.float32, "others":None},
            scale=True
        )
        if tf:
            self.transforms = T.Compose([self.transforms, tf])

    def __getitem__(self, index):
        id, (l,t,w,h) = self.ids_and_windows[index]
        window = win.Window(l,t,w,h)
        path = self.root/self.coco.loadImgs(id)[0]["file_name"]
        with rasterio.open(path, "r") as file:
            image = file.read(window=window, out_dtype=np.uint8) #Window(col_off, row_off, width, height)
        tv_image = tv_tensors.Image(torch.from_numpy(image))
        
        anns = []
        ann_ids = self.coco.getAnnIds(id)
        for ann_id in ann_ids:
            ann = dict(self.coco.anns[ann_id])
            bb_win = win.Window(*ann['bbox'])
            if win.intersect([window, bb_win]):
                x,y,w,h = win.intersection([window, bb_win]).flatten()
                x -= l
                y -= t
                ann['bbox'] = (x,y,w,h)
                anns.append(ann)
        
        #target = self.coco.loadAnns(self.coco.getAnnIds(id))
        target = list_of_dicts_to_dict_of_lists(anns)
        labels = torch.tensor(target["category_id"]) if target else torch.empty(0,)
        boxes = torch.as_tensor(target["bbox"]).float() if target else torch.empty(0,4).float()
        #merged_labels, merged_boxes = merge_labels_boxes(labels, boxes, self.class_list)
        
        merged_labels = []
        merged_boxes = []
        for i, l in enumerate(self.class_list, 1): 
            # indices of tgts whose label belongs to the i-th merge
            idx = reduce(torch.logical_or, [labels == v for v in l.values])
            # i+1 because in detection, class label 0 should be left for no-obj in algos
            merged_labels.append(i * torch.ones_like(labels[idx]))
            merged_boxes.append(boxes[idx])
        merged_labels = torch.cat(merged_labels, dim=0)
        merged_boxes = torch.cat(merged_boxes, dim=0)
        
        tv_target = {}
        tv_target["boxes"] = tv_tensors.BoundingBoxes(
            merged_boxes,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=tuple(T.functional.get_size(tv_image)),
        )
        tv_target['labels'] = merged_labels.long()
        
        if self.transforms is not None:
            tv_image, tv_target = self.transforms(tv_image, tv_target)
        return {'image': tv_image, 'target': tv_target, 'path': path}

    def __len__(self):
        return len(self.ids_and_windows)
    
    @classmethod
    def collate(cls, batch):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        return batch