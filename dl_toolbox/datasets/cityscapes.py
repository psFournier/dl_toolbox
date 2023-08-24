from PIL import Image
import numpy as np
import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms.functional import pil_to_tensor
from dl_toolbox.utils import label, merge_labels


class Cityscapes(Cityscapes):
    # https://pytorch.org/vision/0.15/_modules/torchvision/datasets/cityscapes.html#Cityscapes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_list = [label(
            "void", 
            (0, 0, 0), 
            {c.id for c in self.classes if c.train_id in {255, -1}}
        )]
        self.class_list += [label(c.name, c.color, {c.id}) for c in self.classes if c.train_id not in {255, -1}]
        self.merges = [list(l.values) for l in self.class_list]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")
        image = pil_to_tensor(image) / 255.
        target = Image.open(self.targets[index][0])
        target = np.array(target)
        label = merge_labels(target, self.merges)
        label = torch.from_numpy(label).long()      

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

        image, label = self.transforms(img=image, label=label)

        return {
            "image": image,
            "label": label,
            "path": self.images[index],
        }
