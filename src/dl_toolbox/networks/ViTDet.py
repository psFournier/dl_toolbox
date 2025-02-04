import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelP6P7
from torchvision.ops.misc import Conv2dNormActivation
from collections import OrderedDict

import timm
from typing import Callable, Dict, Optional


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Head(nn.Module):

    def __init__(
        self,
        in_channels,
        n_classes,
        n_share_convs=4,
        n_feat_levels=5
    ):
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

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(n_feat_levels)])

    def forward(self, x):
        cls_logits = []
        bbox_preds = []
        cness_preds = []
        for l, features in enumerate(x):
            features = self.shared_layers(features)
            cls_logits.append(self.cls_logits(features).flatten(-2))
            cness_preds.append(self.ctrness(features).flatten(-2))
            reg = self.bbox_pred(features)
            reg = self.scales[l](reg)
            bbox_preds.append(nn.functional.relu(reg).flatten(-2))
        all_logits = torch.cat(cls_logits, dim=-1).permute(0,2,1) # BxNumAnchorsxC
        all_box_regs = torch.cat(bbox_preds, dim=-1).permute(0,2,1) # BxNumAnchorsx4
        all_cness = torch.cat(cness_preds, dim=-1).permute(0,2,1) # BxNumAnchorsx1
        return all_logits, all_box_regs, all_cness

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return nn.functional.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class SimpleFeaturePyramidNetwork(nn.Module):
    """
    Module that adds a Simple FPN from on top of a set of feature maps. This is based on
    `"Exploring Plain Vision Transformer Backbones for Object Detection" <https://arxiv.org/abs/2203.16527>`_.

    Unlike regular FPN, Simple FPN expects a single feature map,
    on which the Simple FPN will be added.

    Args:
        in_channels (int): number of channels for the input feature map that
            is passed to the module
        out_channels (int): number of channels of the Simple FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: LayerNorm

    Examples::
    
        >>> vitdet = ViTDet(256, 10)
        >>> x = torch.rand(2, 3, 224, 224)
        >>> feat_dict = vitdet.forward_feat(x)
        >>> features = list(feat_dict.values())
        >>> print(f'{[f.shape for f in features] = }')
        >>> box_cls, box_regression, centerness = vitdet.head(features)
        >>> print(f'{box_cls.shape = }')
        >>> assert sum([f.shape[2]*f.shape[3] for f in features])==box_cls.shape[1]

        DOES NOT WORK BELOW
        >>> m = torchvision.ops.SimpleFeaturePyramidNetwork(10, 5)
        >>> # get some dummy data
        >>> x = torch.rand(1, 10, 64, 64)
        >>> # compute the Simple FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_indices=[0,1,2,3],
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.n_feat_levels = 0
        self.strides = []
        for block_index in block_indices:
            layers = []
            current_in_channels = in_channels
            if block_index == 0:
                layers.extend([
                    nn.ConvTranspose2d(
                        in_channels,
                        in_channels // 2,
                        kernel_size=2,
                        stride=2,
                    ),
                    norm_layer(in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        in_channels // 2,
                        in_channels // 4,
                        kernel_size=2,
                        stride=2,
                    ),
                ])
                current_in_channels = in_channels // 4
                stride = 1/4
            elif block_index == 1:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        in_channels // 2,
                        kernel_size=2,
                        stride=2,
                    ),
                )
                current_in_channels = in_channels // 2
                stride = 1/2
            elif block_index == 2:
                # nothing to do for this scale
                stride = 1
                pass
            elif block_index == 3:
                stride = 2
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            layers.extend([
                Conv2dNormActivation(
                    current_in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    norm_layer=norm_layer,
                    activation_layer=None
                ),
                Conv2dNormActivation(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    norm_layer=norm_layer,
                    activation_layer=None
                )
            ])
            self.blocks.append(nn.Sequential(*layers))
            self.n_feat_levels += 1
            self.strides.append(float(stride))

        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}")
        self.extra_blocks = extra_blocks

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Computes the Simple FPN for a feature map.

        Args:
            x (Tensor): input feature map.

        Returns:
            results (list[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        results = [block(x) for block in self.blocks]
        names = [f"{i}" for i in range(len(self.blocks))]

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, [x], names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out
    
class ViTDet(nn.Module):
    
    def __init__(
        self,
        backbone,
        out_channels,
        num_classes,
        add_extra_blocks
    ):
        super(ViTDet, self).__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            dynamic_img_size=True #Deals with inputs of other size than pretraining
        )
        if add_extra_blocks:
            extra_blocks = LastLevelP6P7(
                out_channels,
                out_channels
            )
        else:
            extra_blocks = None
        self.sfpn = SimpleFeaturePyramidNetwork(
            in_channels=self.backbone.embed_dim,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=LayerNorm2d
        )
        n_feat_levels = self.sfpn.n_feat_levels
        patch_size = self.backbone.patch_embed.patch_size[0]
        self.strides = [patch_size*stride for stride in self.sfpn.strides]
        #assert all(map(lambda x: x.is_integer(), strides))
        #self.strides = [int(s) for s in strides]
        if extra_blocks is not None:
            n_feat_levels += 2
        self.head = Head(
            in_channels=out_channels,
            n_classes=num_classes,
            n_feat_levels=n_feat_levels
        )
        
    def forward_feat(self, x):
        intermediates = self.backbone.forward_intermediates(
            x,
            indices=1,
            norm=False,
            intermediates_only=True
        )
        features = self.sfpn(
            intermediates[0]
        )
        return features
    
    def forward(self, x):
        feat_dict = self.forward_feat(x)
        features = list(feat_dict.values())
        box_cls, box_regression, centerness = self.head(features)
        return box_cls, box_regression, centerness