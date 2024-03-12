import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import stochastic_depth


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class ConvNextBlock(nn.Module):

    def __init__(self, filter_dim, kernel_size=7,
                 m=4, layer_scale=1e-6):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(*[
            nn.Conv2d(filter_dim,
                      filter_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=filter_dim),
            Permute([0, 2, 3, 1]),
            LayerNorm(filter_dim, eps=1e-6),
            nn.Linear(filter_dim, filter_dim * m),
            nn.GELU(),
            nn.Linear(filter_dim * m, filter_dim),
            Permute([0, 3, 1, 2])
        ])
        self.gamma = nn.Parameter(torch.ones(filter_dim, 1, 1) * layer_scale)

    def forward(self, x):
        return self.block(x) * self.gamma


class ConvNextLayer(nn.Module):

    def __init__(self, filter_dim, depth, drop_rates):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(ConvNextBlock(filter_dim=filter_dim))

        self.drop_rates = drop_rates

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = x + stochastic_depth(block(x),
                                     self.drop_rates[idx],
                                     mode="batch",
                                     training=self.training)
        return x


class ConvNext(nn.Module):

    def __init__(self,
                 num_channels=3,
                 patch_size=4,
                 layer_dims=[96, 192, 384, 768],
                 depths=[3, 3, 9, 3],
                 drop_rate=0.):
        super().__init__()

        # init downsample layers with stem
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(num_channels, layer_dims[0], kernel_size=patch_size, stride=patch_size),
                LayerNorm(layer_dims[0],
                              eps=1e-6,
                              data_format="channels_first")
            )])
        for idx in range(len(layer_dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(layer_dims[idx],
                              eps=1e-6,
                              data_format="channels_first"),
                    nn.Conv2d(layer_dims[idx],
                              layer_dims[idx + 1],
                              kernel_size=2,
                              stride=2),
                ))

        drop_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
        self.stage_layers = nn.ModuleList([])
        for idx, layer_dim in enumerate(layer_dims):
            layer_dr = drop_rates[sum(depths[:idx]): sum(depths[:idx]) + depths[idx]]
            self.stage_layers.append(
                ConvNextLayer(filter_dim=layer_dim, depth=depths[idx], drop_rates=layer_dr))


    def forward(self, x):
        outputs = []
        all_layers = list(zip(self.downsample_layers, self.stage_layers))
        for downsample_layer, stage_layer in all_layers:
            x = downsample_layer(x)
            x = stage_layer(x)
            outputs.append(x)
        # we want only last three feature maps (C3, C4, C5)
        return outputs[1:]



def _group_norm(out_channels, groups=32, affine=True, epsilon=1e-5):
    return torch.nn.GroupNorm(groups, out_channels, epsilon, affine)


def _conv_block(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    conv = nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=dilation * (kernel_size - 1) // 2,
                     dilation=dilation,
                     bias=False)
    return nn.Sequential(conv, _group_norm(out_channels), nn.ReLU(inplace=True))


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block_module = _conv_block(in_channels,
                                             out_channels,
                                             kernel_size=1,
                                             stride=1)
            layer_block_module = _conv_block(out_channels,
                                             out_channels,
                                             kernel_size=3,
                                             stride=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        self.top_blocks = LastLevelP6P7(out_channels, out_channels)

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            inner_top_down = F.interpolate(last_inner,
                                           size=(int(inner_lateral.shape[-2]),
                                                 int(inner_lateral.shape[-1])),
                                           mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))

        last_results = self.top_blocks(x[-1], results[-1])
        results.extend(last_results)

        return tuple(results)


class LastLevelP6P7(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Head(nn.Module):

    def __init__(self, in_channels, n_classes, n_share_convs=4):
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

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        cls_logits = []
        bbox_preds = []
        cness_preds = []
        for l, features in enumerate(x):
            features = self.shared_layers(features)

            cls_logits.append(self.cls_logits(features))
            cness_preds.append(self.ctrness(features))
            reg = self.bbox_pred(features)
            reg = self.scales[l](reg)
            bbox_preds.append(F.relu(reg))
        return cls_logits, bbox_preds, cness_preds


class FCOS(torch.nn.Module):

    def __init__(self,
                 in_channels=[192, 384, 768],
                 out_channels=192,
                 num_classes=19,
                 backbone_layer_dims=[96, 192, 384, 768],
                 backbone_depths=[3, 9, 3, 3]):
        super(FCOS, self).__init__()

        backbone = ConvNext(num_channels=3,
                            patch_size=4,
                            layer_dims=backbone_layer_dims,
                            depths=backbone_depths,
                            drop_rate=0.0)

        fpn = FeaturePyramidNetwork(in_channels_list=in_channels,
                                    out_channels=out_channels)
        self.feature_extractor = nn.Sequential(backbone, fpn)
        self.head = Head(out_channels, num_classes)

    def forward(self, images):
        features = self.feature_extractor(images)
        box_cls, box_regression, centerness = self.head(features)
        return features, box_cls, box_regression, centerness