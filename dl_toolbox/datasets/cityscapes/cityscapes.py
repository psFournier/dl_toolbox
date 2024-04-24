from PIL import Image
import numpy as np
import torch
import enum
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms.functional import pil_to_tensor
from dl_toolbox.utils import label, merge_labels

void_cls = {c.id for c in Cityscapes.classes if c.train_id in {255, -1}}
all19 = [label("void", (0, 0, 0), void_cls)] + [
    label(c.name, c.color, {c.id}) for c in Cityscapes.classes if c.train_id not in {255, -1}
]

classes = enum.Enum(
    "CityscapesClasses",
    {
        "all19": all19,
    },
)

class Cityscapes(Dataset):
    # https://pytorch.org/vision/0.15/_modules/torchvision/datasets/cityscapes.html#Cityscapes
    
    classes = classes

    def __init__(self, imgs, msks, merge, transforms):
        self.imgs = imgs
        self.msks = msks
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.imgs[index]).convert("RGB")
        image = pil_to_tensor(image)
        label = None
        if self.msks:
            target = Image.open(self.msks[index])
            target = np.array(target)
            label = merge_labels(target, self.merges)
            label = torch.from_numpy(label).long().unsqueeze(0)    

        #targets = []
        #for i, t in enumerate(self.target_type):
        #    if t == "polygon":
        #        target = self._load_json(self.targets[index][i])
        #    else:
        #        target = Image.open(self.targets[index][i])
        #        target = pil_to_tensor(target).squeeze()
#
        #    targets.append(target)
#
        #target = tuple(targets) if len(targets) > 1 else targets[0]
        if self.transforms is not None:
            image, label = self.transforms(img=image, label=label)

        return {
            "image": image,
            "label": None if label is None else label.squeeze(),
            "image_path": self.imgs[index],
            #"window": win,
            "label_path": None if label is None else self.msks[index]
        }
