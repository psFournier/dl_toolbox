from segmentation_models_pytorch import Unet


class SmpUnet(Unet):
    def __init__(
        self,
        num_classes,
        *args,
        **kwargs
    ):
        super().__init__(
            classes=num_classes,
            *args, **kwargs
        )