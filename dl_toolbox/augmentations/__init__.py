from .utils import Compose, rand_bbox, NoOp, OneOf
from .color import Saturation, Contrast, Brightness, Gamma, Color
from .d4 import Vflip, Hflip, Transpose1, Transpose2, D4, Rot90, Rot270, Rot180
from .geometric import Sharpness
from .histograms import HistEq
from .mixup import Mixup, Mixup2
from .cutmix import Cutmix, Cutmix2
from .merge_label import MergeLabels
from .normalizations import ImagenetNormalize
from .crop import RandomCrop2

aug_dict = {
    'no': NoOp,
    'imagenet': ImagenetNormalize, 
    'd4': D4,
    'hflip': Hflip,
    'vflip': Vflip,
    'd1flip': Transpose1,
    'd2flip': Transpose2,
    'rot90': Rot90,
    'rot180': Rot180,
    'rot270': Rot270,
    'saturation': Saturation,
    'sharpness': Sharpness,
    'contrast': Contrast,
    'gamma': Gamma,
    'brightness': Brightness,
    'color': Color,
    'cutmix': Cutmix,
    'mixup': Mixup
}

anti_aug_dict = {
    'no': NoOp,
    'imagenet': NoOp, 
    'hflip': Hflip,
    'vflip': Vflip,
    'd1flip': Transpose1,
    'd2flip': Transpose2,
    'rot90': Rot270,
    'rot180': Rot180,
    'rot270': Rot90,
    'saturation': NoOp,
    'sharpness': NoOp,
    'contrast': NoOp,
    'gamma': NoOp,
    'brightness': NoOp,
}

def get_transforms(name):
    
    if name:
        parts = name.split('_')
        aug_list = []
        for part in parts:
            if part.startswith('color'):
                bounds = part.split('-')[-1]
                augment = Color(bound=0.1*int(bounds))
            elif part.startswith('cutmix2'):
                alpha = part.split('-')[-1]
                augment = Cutmix(alpha=0.1*int(alpha))
            else:
                augment = aug_dict[part]()
            aug_list.append(augment)
        return Compose(aug_list)
    else:
        return NoOp()
