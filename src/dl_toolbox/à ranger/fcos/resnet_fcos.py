from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
import torch.nn as nn
import torch
import math

class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Head(nn.Module):

    def __init__(self, in_channels, n_classes, n_share_convs=4, n_feat_levels=5):
        super().__init__()

        #tower = []
        cls_tower = []
        bbox_tower = []
        for _ in range(n_share_convs):
            cls_tower.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
        self.cls_layers = nn.Sequential(*cls_tower)
        self.bbox_layers = nn.Sequential(*bbox_tower)

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
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(n_feat_levels)])

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_logits = []
        bbox_preds = []
        cness_preds = []
        for l, features in enumerate(x):
            cls_features = self.cls_layers(features) # BxinChannelsxfeatSize
            bbox_features = self.bbox_layers(features)
            cls_logits.append(self.cls_logits(cls_features).flatten(-2)) # BxNumClsxFeatSize
            cness_preds.append(self.ctrness(bbox_features).flatten(-2)) # Bx1xFeatSize
            reg = self.bbox_pred(bbox_features) # Bx4xFeatSize
            reg = self.scales[l](reg)
            bbox_preds.append(torch.exp(reg).flatten(-2))
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
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2), 
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
        inp = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            out = self.features(inp)
        # feat_sizes is the list of feature size of each stage of the FPN
        self.feat_sizes = [o.shape[2:] for o in out.values()]
        print(self.feat_sizes)
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