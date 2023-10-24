import pytorch_lightning as pl
import torch
import torch.nn as nn
from .supervised import Supervised


class SelfTraining(Supervised):
    def __init__(
        self,
        strong_tf,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.strong_tf = strong_tf

    def forward(self, x):
        return self.network.forward(self.norm(x))

    def training_step(self, batch, batch_idx):
        batch, pseudosup_batch = batch["sup"], batch["pseudosup"]
        xs = batch["image"]
        ys = batch["label"]
        ys_o = self.one_hot(ys)
        xs, ys_o = self.batch_tf(xs, ys_o)
        # ST    
        x_pl = pseudosup_batch["image"]
        prob_y_pl = pseudosup_batch["label"]
        _, y_pl = torch.max(prob_y_pl, dim=1)
        x_pl_strong, y_pl_strong = self.strong_tf(x_pl, y_pl)
        x = torch.vstack((xs_weak, x_pl_strong))
        y = torch.vstack((ys_weak, y_pl_strong))
        logits_x = self.forward(x)
        ce = self.ce(logits_x, y)
        self.log(f"cross_entropy/train", ce)
        dice = 0
        if self.dice is not None: 
            dice = self.dice(logits_x, ys_o)
            self.log(f"dice/train", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits_x))
        self.train_accuracy.update(preds, y)
        return ce + dice