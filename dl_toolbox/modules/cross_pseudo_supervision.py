import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.losses import DiceLoss, ProbOhemCrossEntropy2d
from dl_toolbox.utils import plot_confusion_matrix
from .supervised import Supervised


class CrossPseudoSupervision(Supervised):
    def __init__(
        self,
        network2,
        alpha_ramp,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.network1 = self.network
        self.network2 = network2(
            in_channels=kwargs['in_channels'], num_classes=kwargs['num_classes']
        )
        self.alpha_ramp = alpha_ramp

    def forward(self, x):
        return self.network1(x)

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)

    def training_step(self, batch, batch_idx):
        batch, unsup_batch = batch["sup"], batch["unsup"]
        xs = batch["image"]
        ys = batch["label"]
        xs, ys = self.tf(xs, ys)
        logits1 = self.network1(xs)
        logits2 = self.network2(xs)
        ce = self.ce(logits1, ys) + self.ce(logits2, ys)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits1, ys)+self.dice(logits2, ys)
        self.log(f"dice/train", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits1))
        self.train_accuracy.update(preds, ys)
            
        unsup_inputs = unsup_batch["image"]
        unsup_logits_1 = self.network1(unsup_inputs)
        unsup_logits_2 = self.network2(unsup_inputs)
        with torch.no_grad():
            _, sup_pl_2 = torch.max(logits2, dim=1)
            _, unsup_pl_2 = torch.max(unsup_logits_2, dim=1)
            _, sup_pl_1 = torch.max(logits1, dim=1)
            _, unsup_pl_1 = torch.max(unsup_logits_1, dim=1)
        cps_loss_unlabeled = self.ce(unsup_logits_1, unsup_pl_2) + self.ce(unsup_logits_2, unsup_pl_1)
        self.log("cps_loss_unlabeled/train", cps_loss_unlabeled)
        cps_loss_labeled = self.ce(logits1, sup_pl_2) + self.ce(logits2, sup_pl_1)
        self.log("cps_loss_labeled/train", cps_loss_labeled)
        
        pl_acc = sup_pl_1.eq(ys).float().mean()
        self.log("pseudolabel accuracy/train", pl_acc)
    
        return self.ce_weight * ce + self.dice_weight * dice + self.alpha * (cps_loss_labeled + cps_loss_unlabeled)