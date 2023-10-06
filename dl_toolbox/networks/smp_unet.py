from segmentation_models_pytorch import Unet


class SmpUnet(Unet):
    def __init__(
        self,
        num_classes,
        **kwargs
    ):
        super().__init__(
            classes=num_classes,
            **kwargs
        )