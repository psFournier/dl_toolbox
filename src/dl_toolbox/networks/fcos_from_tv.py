from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from dl_toolbox.networks.fcos import Head
import torch.nn as nn
import torch

class FCOS(nn.Module):
    
    def __init__(self, image_size, num_classes, in_channels=3, out_channels=256):
        super(FCOS, self).__init__()
        #backbone = resnet50(weights=None)#ResNet50_Weights.DEFAULT)
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        return_nodes = {
            'features.3.2.add': 'layer2',
            'features.5.8.add': 'layer3',
            'features.7.2.add': 'layer4',
        }
        # Extract 3 main layers
        self.feature_extractor = create_feature_extractor(backbone, return_nodes)
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, image_size, image_size)
        with torch.no_grad():
            out = self.feature_extractor(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels,out_channels)
        )
        self.fpn_features = nn.Sequential(self.feature_extractor, fpn)
        inp = torch.randn(2, 3, image_size, image_size)
        with torch.no_grad():
            out = self.fpn_features(inp)
        self.feat_sizes = [o.shape[2:] for o in out.values()]
        self.head = Head(out_channels, num_classes)

    def forward(self, images):
        features = list(self.fpn_features(images).values())
        box_cls, box_regression, centerness = self.head(features)
        all_level_preds = (torch.cat([t.flatten(-2) for t in o], dim=-1) for o in [features, box_cls, box_regression, centerness])
        return (torch.permute(t, (0,2,1)) for t in all_level_preds)