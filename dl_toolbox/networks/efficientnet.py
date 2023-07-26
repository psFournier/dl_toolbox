from ._efficientnet import EfficientNet, _efficientnet_conf
import torch.nn as nn


class EfficientNet_b0(EfficientNet):
    def __init__(
        self,
        in_channels,
        num_classes,
        **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            "efficientnet_b0", width_mult=1.0, depth_mult=1.0
        )
        super().__init__(
            inverted_residual_setting,
            kwargs.pop("dropout", 0.2),
            last_channel=last_channel,
            num_classes=num_classes,
            **kwargs
        )
        assert in_channels==3