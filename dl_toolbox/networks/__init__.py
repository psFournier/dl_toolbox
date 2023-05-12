from .unet import Unet
from .smp_unet import SmpUnet
from .vgg import Vgg
from .flair import *
from .encode_then_upsample import *
from .efficientnet import *

networks = {
    'Vgg': Vgg,
    'Unet': Unet,
    'SmpUnet': SmpUnet,
    'EncodeThenUpsample': EncodeThenUpsample,
    'Efficientnetb0': efficientnet_b0
}

class NetworkFactory:

    @staticmethod
    def create(name):
        return networks[name]
