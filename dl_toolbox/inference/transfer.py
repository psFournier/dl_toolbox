import torch 
from collections import OrderedDict
from omegaconf import OmegaConf
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import os

    
class Class(NamedTuple):
    name: str
    id: int
    background: bool = False
    color: Optional[Tuple] = None
    category: Optional[str] = None
    
def to_dict(train_classes) -> Dict[int, Any]:
    """Export class manager to dictionnary."""
    classes_dict: Dict[int, Any] = {}
    for _class in train_classes:
        classes_dict[_class.id] = {'name': _class.name}
        if _class.color is not None:
            classes_dict[_class.id]['color'] = tuple(_class.color)
    return classes_dict

class_colors = {
    "other": (0, 0, 0), 
    "bareground": (100, 50, 0),
    "low_vegetation": (0, 250, 50),
    "water": (0, 50, 250),
    "building": (250, 50, 50), 
    "high_vegetation": (0, 100, 50),
    "parking": (200,200,200),
    "road": (100, 100, 100), 
    "railways": (200,100,200),
    "swimmingpool": (50,150,250),
}



ckpt_path = '/work/OT/ai4usr/fournip/outputs/digitaniev2/ce_d4color3/05Jan23-17h54/checkpoints/epoch=428-step=133847.ckpt'
ckpt = torch.load(ckpt_path)

metadata_conf = OmegaConf.create()
metadata_conf['model'] = {'_target_': 'segmentation_models_pytorch.Unet', 'encoder_name': ckpt['hyper_parameters']['encoder'], 'encoder_weights': 'imagenet', 'in_channels': ckpt['hyper_parameters']['in_channels'], 'classes': ckpt['hyper_parameters']['out_channels']}


train_classes = []
for i, class_name in enumerate(ckpt['hyper_parameters']['class_names']):
    train_classes.append(Class(name=class_name, id=i, color=class_colors[class_name]))
# Get class manager information
metadata_conf['classes'] = to_dict(train_classes)

output_path = '/work/OT/ai4usr/hummerc/dlcooker_wrapper/'
state_dict = OrderedDict()
for key, value in ckpt['state_dict'].items():
    if not 'loss.weight' in key:
        state_dict[key.replace('network.', '')] = value

# Export state_dict-only model in PyTorch format
torch.save(state_dict, os.path.join(output_path, 'model.pth'))
OmegaConf.save(metadata_conf, os.path.join(output_path, 'metadata.yaml'))