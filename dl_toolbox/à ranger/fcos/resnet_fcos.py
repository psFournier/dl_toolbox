from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
import torch.nn as nn
import torch
import torch.nn.functional as F

class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Head(nn.Module):

    def __init__(self, in_channels, n_classes, n_share_convs=4, n_feat_levels=5):
        super().__init__()

        tower = []
        for _ in range(n_share_convs):
            tower.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            tower.append(nn.GroupNorm(32, in_channels))
            tower.append(nn.ReLU())
        self.shared_layers = nn.Sequential(*tower)

        self.cls_logits = nn.Conv2d(in_channels,
                                    n_classes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.ctrness = nn.Conv2d(in_channels,
                                 1,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        
        # What is this ?
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(n_feat_levels)])

    def forward(self, x):
        cls_logits = []
        bbox_preds = []
        cness_preds = []
        for l, features in enumerate(x):
            features = self.shared_layers(features) # BxinChannelsxfeatSize
            cls_logits.append(self.cls_logits(features).flatten(-2)) # BxNumClsxFeatSize
            cness_preds.append(self.ctrness(features).flatten(-2)) # Bx1xFeatSize
            reg = self.bbox_pred(features) # Bx4xFeatSize
            reg = self.scales[l](reg)
            bbox_preds.append(F.relu(reg).flatten(-2))
        all_logits = torch.cat(cls_logits, dim=-1).permute(0,2,1) # BxNumAnchorsxC
        all_box_regs = torch.cat(bbox_preds, dim=-1).permute(0,2,1) # BxNumAnchorsx4
        all_cness = torch.cat(cness_preds, dim=-1).permute(0,2,1) # BxNumAnchorsx1
        return all_logits, all_box_regs, all_cness

class ResnetDet(nn.Module):
    def __init__(self, out_channels, num_classes):
        super().__init__()
        # Defines and specifies node names for feature extraction model
        # In resnet50 layerN is conv{N+1}_x of the resnet paper
        # In FPN paper, conv2_x output is called C2
        # Here we extract only C3, C4 and C5 following retinaNet
        # conv1 of resnet50 div by 2 feat maps, layer1 by 2, layer2 by 2 
        self.backbone = create_feature_extractor(
            resnet50(), 
            {
                'layer2.3.relu_2': 'layer2', # 1/8th feat map
                'layer3.5.relu_2': 'layer3', # 1/16
                'layer4.2.relu_2': 'layer4', # 1/32
            }
        )
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN P5, P4, P3 (no P2 following retinanet)
        # Following RetinaNet, add P6 and P7 to detect bigger objects
        # Necessary for teledec ? not sure.
        fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels,out_channels)
        )
        self.features = nn.Sequential(self.backbone, fpn)
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.features(inp)
        # feat_sizes is the list of feature size of each stage of the FPN
        self.feat_sizes = [o.shape[2:] for o in out.values()]
        self.head = Head(out_channels, num_classes)
        
    def forward(self, x):
        """
        returns
            box_cls: list of logits of sizes BxNumClsxFeatSize
            box_reg: list of pred bb of sizes Bx4xFeatSize
            centerness: list of Bx1xFeatSize ?
        """
        features = list(self.features(x).values()) # feature maps from FPN
        box_cls, box_regression, centerness = self.head(features)
        return box_cls, box_regression, centerness