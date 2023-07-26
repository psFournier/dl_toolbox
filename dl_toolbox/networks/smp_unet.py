from segmentation_models_pytorch import Unet


class SmpUnet(Unet):
    def __init__(
        self,
        in_channels,
        num_classes,
        *args,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            classes=num_classes,
            *args, **kwargs
        )