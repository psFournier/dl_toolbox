from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from pathlib import Path
import torch
import torchvision.transforms.v2 as v2
import rasterio
import numpy as np
from torchvision.io import read_image

from dl_toolbox.utils import label, merge_labels

CAT_ID_TO_NAME = {
    0: u'__background__',
    1: u'person',
    2: u'bicycle',
    3: u'car',
    4: u'motorbike',
    5: u'aeroplane',
    6: u'bus',
    7: u'train',
    8: u'truck',
    9: u'boat',
    10: u'trafficlight',
    11: u'firehydrant',
    12: u'streetsign',
    13: u'stopsign',
    14: u'parkingmeter',
    15: u'bench',
    16: u'bird',
    17: u'cat',
    18: u'dog',
    19: u'horse',
    20: u'sheep',
    21: u'cow',
    22: u'elephant',
    23: u'bear',
    24: u'zebra',
    25: u'giraffe',
    26: u'hat',
    27: u'backpack',
    28: u'umbrella',
    29: u'shoe',
    30: u'eyeglasses',
    31: u'handbag',
    32: u'tie',
    33: u'suitcase',
    34: u'frisbee',
    35: u'skis',
    36: u'snowboard',
    37: u'sportsball',
    38: u'kite',
    39: u'baseballbat',
    40: u'baseballglove',
    41: u'skateboard',
    42: u'surfboard',
    43: u'tennisracket',
    44: u'bottle',
    45: u'plate',
    46: u'wineglass',
    47: u'cup',
    48: u'fork',
    49: u'knife',
    50: u'spoon',
    51: u'bowl',
    52: u'banana',
    53: u'apple',
    54: u'sandwich',
    55: u'orange',
    56: u'broccoli',
    57: u'carrot',
    58: u'hotdog',
    59: u'pizza',
    60: u'donut',
    61: u'cake',
    62: u'chair',
    63: u'sofa',
    64: u'pottedplant',
    65: u'bed',
    66: u'mirror',
    67: u'diningtable',
    68: u'window',
    69: u'desk',
    70: u'toilet',
    71: u'door',
    72: u'tvmonitor',
    73: u'laptop',
    74: u'mouse',
    75: u'remote',
    76: u'keyboard',
    77: u'cellphone',
    78: u'microwave',
    79: u'oven',
    80: u'toaster',
    81: u'sink', 
    82: u'refrigerator',
    83: u'blender',
    84: u'book',
    85: u'clock',
    86: u'vase',
    87: u'scissors',
    88: u'teddybear',
    89: u'hairdrier',
    90: u'toothbrush',
    91: u'hairbrush'
}

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = defaultdict(list)
    for dct in list_of_dicts:
        for key, value in dct.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)

class Coco(Dataset):
    
    classes = [label(v, (0, 255, 255), {k}) for k, v in CAT_ID_TO_NAME.items()]

    def __init__(self, root, annFile, transforms):
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

    def __getitem__(self, index):
        id = self.ids[index]
        path = self.root/self.coco.loadImgs(id)[0]["file_name"]
        tv_image = tv_tensors.Image(read_image(path))
        
        target = self.coco.loadAnns(self.coco.getAnnIds(id))
        target = list_of_dicts_to_dict_of_lists(target)
        tv_target = {}
        tv_target["boxes"] = tv_tensors.BoundingBoxes(
            target["bbox"],
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=tuple(F.get_size(tv_image)),
        )
        labels = torch.tensor(target["category_id"])
        tv_target['labels'] = labels.long()
        if self.transforms is not None:
            tv_image, tv_target = self.transforms(tv_image, tv_target)
        return tv_image, tv_target, path

    def __len__(self):
        return len(self.ids)