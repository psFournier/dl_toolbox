from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from collections import namedtuple

labels = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'] 

label = namedtuple("label", ["name", "color", "values"])
all20 = [label(l, (255,0,0), {i}) for i, l in enumerate(labels)]

class PascalVOC(Dataset):
    
    classes = {
        "all20": all20,
    }

    def __init__(self, data, transforms=None):
        image_paths = []
        targets = []
        for instance in data:
            image_paths.append(instance['image_path'])
            targets.append(instance["target"])
        self.image_paths = image_paths
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        image = v2.functional.pil_to_tensor(image)
        targets = self.targets[idx]
        targets = torch.Tensor(targets)
        bboxes = tv_tensors.BoundingBoxes(targets[:,:4], format="XYXY", canvas_size=(h,w))
        labels = targets[:, 4:]
        if self.transforms:
            image, bboxes = self.transforms(image, bboxes)
        return image, bboxes, labels, image_path