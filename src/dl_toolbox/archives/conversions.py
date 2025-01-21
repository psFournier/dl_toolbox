import torch
import torchvision.transforms.v2 as v2

class ToFloat32(v2.ToDtype):
    def __init__(self, scale=True):
        super().__init__(dtype=torch.float32, scale=scale)