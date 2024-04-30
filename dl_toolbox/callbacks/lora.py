from pytorch_lightning.callbacks import BaseFinetuning
from functools import partial
import minlora 
import torch.nn as nn

class Lora(BaseFinetuning):
    def __init__(self,  module_name, rank):
        super().__init__()
        self.module_name = module_name
        self.cfg = self.get_lora_config(rank)
        
    def get_lora_config(self, rank):
        return {  # specify which layers to add lora to, by default only add to linear layers
            nn.Linear: {
                "weight": partial(minlora.LoRAParametrization.from_linear, rank=rank),
            },
        }
        
    def freeze_before_training(self, pl_module):
        self.freeze(getattr(pl_module, self.module_name), train_bn=True)
        minlora.add_lora(getattr(pl_module, self.module_name), lora_config=self.cfg)
        
    def finetune_function(self, pl_module, current_epoch, optimizer):
        pass