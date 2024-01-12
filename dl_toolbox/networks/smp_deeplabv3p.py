from segmentation_models_pytorch import DeepLabV3Plus


class SmpDeeplabV3p(DeepLabV3Plus):
    def __init__(
        self,
        num_classes,
        **kwargs
    ):
        super().__init__(
            classes=num_classes,
            **kwargs
        )
        self.feature_extractor = self.encoder.get_stages()