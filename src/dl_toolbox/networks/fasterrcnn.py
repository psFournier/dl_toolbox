import torch.nn as nn
import math


class FasterRCNN(EfficientNet):
    def __init__(
        self,
        in_channels,
        num_classes,
        weights=None,
        norm_layer=None,
        **kwargs
    ):  
        weights = EfficientNet_B0_Weights[weights] if weights else None
        out_channels = len(weights.meta["categories"]) if weights else num_classes
        assert in_channels==3
        inverted_residual_setting, last_channel = _efficientnet_conf(
            "efficientnet_b0", width_mult=1.0, depth_mult=1.0
        )
        super().__init__(
            inverted_residual_setting,
            kwargs.pop("dropout", 0.2),
            last_channel=last_channel,
            num_classes=out_channels,
            norm_layer=norm_layer,
            **kwargs
        )

        if weights is not None:
            self.load_state_dict(
                weights.get_state_dict(progress=True)
            )
            
            # switching head for num_class and init
            head = nn.Linear(1280, num_classes)  
            self.classifier[-1] = head
            init_range = 1.0 / math.sqrt(num_classes)
            nn.init.uniform_(head.weight, -init_range, init_range)
            nn.init.zeros_(head.bias)
            
        self.feature_extractor = self.features
