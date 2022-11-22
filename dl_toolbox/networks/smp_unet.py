from argparse import ArgumentParser
import segmentation_models_pytorch as smp

class SmpUnet(smp.Unet):

    def __init__(
        self,
        encoder,
        pretrained,
        in_channels,
        out_channels,
        *args,
        **kwargs
    ):
        super().__init__(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
            decoder_use_batchnorm=True
        )
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.out_dim = (1,1)

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--out_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        
        return parser
