from transformers import ViTForImageClassification
import torch.nn as nn
import math

class ViT(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        weights,
        norm_layer=None,
        **kwargs
    ):  
        super(ViT,self).__init__(**kwargs)
        assert in_channels==3
        self.weights = weights
        self.num_classes = num_classes
        self.model = ViTForImageClassification.from_pretrained(weights, num_labels=num_classes)
        self.feature_extractor = self.model.vit

    def forward(self, x):
        outputs = self.model.forward(pixel_values=x, return_dict=False)
        return outputs[0]