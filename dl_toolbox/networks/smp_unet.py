from argparse import ArgumentParser

import segmentation_models_pytorch as smp


class SmpUnet(smp.Unet):
    def __init__(
        self,
        encoder,
        in_channels,
        out_channels,
        pretrained=False,
        bn=True,
        *args,
        **kwargs
    ):
        super().__init__(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
            decoder_use_batchnorm=bn,
        )
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.out_dim = (1, 1)
