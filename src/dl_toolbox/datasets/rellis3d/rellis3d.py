import enum
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2 
from torchvision import tv_tensors

from dl_toolbox.utils import merge_labels, label

onto0 = {0: 'void', 1: 'dirt', 3: 'grass', 4: 'tree', 5: 'pole', 6: 'water', 7: 'sky', 8: 'vehicle', 9: 'object', 10: 'asphalt', 12: 'building', 15: 'log', 17: 'person', 18: 'fence', 19: 'bush', 23: 'concrete', 27: 'barrier', 31: 'puddle', 33: 'mud', 34: 'rubble'}

onto1 = {0: [0, 0, 0], 1: [108, 64, 20], 3: [0, 102, 0], 4: [0, 255, 0], 5: [0, 153, 153], 6: [0, 128, 255], 7: [0, 0, 255], 8: [255, 255, 0], 9: [255, 0, 127], 10: [64, 64, 64], 12: [255, 0, 0], 15: [102, 0, 0], 17: [204, 153, 255], 18: [102, 0, 204], 19: [255, 153, 204], 23: [170, 170, 170], 27: [41, 121, 255], 31: [134, 255, 239], 33: [99, 66, 34], 34: [110, 22, 138]}

all20 = [label(v, tuple(onto1[k]), {k}) for k, v in onto0.items()]

class Rellis3d(Dataset):
    
    all_class_lists = enum.Enum(
        "Rellis3dClasses",
        {
            "all20": all20,
        },
    )

    def __init__(self, imgs, msks, merge, transforms):
        self.imgs = imgs
        self.msks = msks
        self.transforms = v2.ToDtype(dtype={
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.int64,
            "others":None
        }, scale=True)
        if transforms is not None:
            self.transforms = v2.Compose([self.transforms, transforms])
        self.merges = None
        if merge:
            self.class_list = self.all_class_lists[merge].value
            self.merges = [list(l.values) for l in self.class_list]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        image = tv_tensors.Image(read_image(image_path))
        target = None
        if self.msks:
            mask = read_image(self.msks[idx])
            if self.merges:
                mask = merge_labels(mask, self.merges)
            target = tv_tensors.Mask(mask)
        image, target = self.transforms(image, target)
        if self.msks:
            target = target.squeeze()
        return {'image':image, 'target':target, 'image_path':image_path}