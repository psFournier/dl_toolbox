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
        sup_loss = super().training_step(batch, batch_idx)      
        #CPS
        unsup_inputs = batch["unsup"]["image"]
        unsup_logits_1 = self.network1(unsup_inputs)
        unsup_logits_2 = self.network2(unsup_inputs)
        with torch.no_grad():
            _, unsup_pl_2 = torch.max(unsup_logits_2, dim=1)
            _, unsup_pl_1 = torch.max(unsup_logits_1, dim=1)
        cps_loss = self.ce(unsup_logits_1, unsup_pl_2) + self.ce(unsup_logits_2, unsup_pl_1)
        self.log("cps_loss/train", cps_loss)
        if batch["unsup"]["label"] is not None:
            pl_acc = unsup_pl_1.eq(batch["unsup"]["label"]).float().mean()
            self.log("pseudolabel accuracy/train", pl_acc)
        return sup_loss + self.alpha * cps_loss