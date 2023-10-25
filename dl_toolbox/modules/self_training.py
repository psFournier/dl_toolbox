import pytorch_lightning as pl
import torch
import torch.nn as nn
from .supervised import Supervised


class SelfTraining(Supervised):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        batch, pseudosup_batch = batch["sup"], batch["pseudosup"]
        xs = batch["image"]
        ys = batch["label"]
        ys_o = self.one_hot(ys)
        xs, ys_o = self.batch_tf(xs, ys_o)
        # ST    
        x_pl = pseudosup_batch["image"]
        y_pl = pseudosup_batch["label"]
        y_pl_o = self.one_hot(y_pl)
        x = torch.vstack((xs, x_pl))
        y = torch.vstack((ys_o, y_pl_o))
        logits_x = self.forward(x)
        ce = self.ce(logits_x, y)
        self.log(f"cross_entropy/train", ce)
        dice = 0
        if self.dice is not None: 
            dice = self.dice(logits_x, y)
            self.log(f"dice/train", dice)
        return ce + dice