from .color import Brightness, Color, Contrast, Gamma, Saturation
from .crop import RandomCrop2, RandomResizedCrop, RemovePad, PadSymmetric
from .cutmix import Cutmix, Cutmix2
from .d4 import D4, Flip, Hflip, Rot180, Rot270, Rot90, Transpose1, Transpose2, Vflip
from .mixup import Mixup, Mixup2
from .resize import Resize
from .utils import Compose, NoOp, OneOf, rand_bbox, TTA, Sliding
from .imagenet import *
from .stretch_to_minmax import *
from .zero_average import *
from .conversions import ToFloat32

aug_dict = {
    "no": NoOp,
    "d4": D4,
    "flip": Flip,
    "hflip": Hflip,
    "vflip": Vflip,
    "d1flip": Transpose1,
    "d2flip": Transpose2,
    "rot90": Rot90,
    "rot180": Rot180,
    "rot270": Rot270,
    "resize": Resize,
    "saturation": Saturation,
    "contrast": Contrast,
    "gamma": Gamma,
    "brightness": Brightness,
    "color": Color,
    "cutmix": Cutmix,
    "mixup": Mixup,
}

anti_aug_dict = {
    "no": NoOp,
    "imagenet": NoOp,
    "hflip": Hflip,
    "vflip": Vflip,
    "d1flip": Transpose1,
    "d2flip": Transpose2,
    "rot90": Rot270,
    "rot180": Rot180,
    "rot270": Rot90,
    "saturation": NoOp,
    "sharpness": NoOp,
    "contrast": NoOp,
    "gamma": NoOp,
    "brightness": NoOp,
}


def get_transforms(name):
    if name:
        parts = name.split("_")
        aug_list = []
        for part in parts:
            if part.startswith("color"):
                bounds = part.split("-")[-1]
                augment = Color(bound=0.1 * int(bounds))
            elif part.startswith("cutmix2"):
                alpha = part.split("-")[-1]
                augment = Cutmix(alpha=0.1 * int(alpha))
            else:
                augment = aug_dict[part]()
            aug_list.append(augment)
        return Compose(aug_list)
    else:
        return NoOp()
