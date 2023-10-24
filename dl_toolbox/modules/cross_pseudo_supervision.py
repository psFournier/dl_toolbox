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
        cutmix,
        alpha_ramp,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.network1 = self.network
        self.network2 = network2(
            in_channels=kwargs['in_channels'], num_classes=kwargs['num_classes']
        )
        self.cutmix = cutmix
        self.alpha_ramp = alpha_ramp

    def forward(self, x):
        return self.network1.forward(self.norm(x))

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)

    def training_step(self, batch, batch_idx):
        sup_loss = super().training_step(batch, batch_idx)      
        #CPS
        xu = batch["unsup"]["image"]
        with torch.no_grad():
            yu1 = self.network1.forward(self.norm(xu))
            _, yu1 = torch.max(yu1, dim=1)
            yu2 = self.network2.forward(self.norm(xu))
            _, yu2 = torch.max(yu2, dim=1)
        xu1, yu1 = self.cutmix(xu, yu1)
        xu2, yu2 = self.cutmix(xu, yu2)
        logits_xu1 = self.network1.forward(self.norm(xu1))
        logits_xu2 = self.network2.forward(self.norm(xu2))
        cps_loss = self.ce(logits_xu1, yu2) + self.ce(logits_xu2, yu1)
        self.log("cps_loss/train", cps_loss)
        if batch["unsup"]["label"] is not None:
            pl_acc = unsup_pl_1.eq(batch["unsup"]["label"]).float().mean()
            self.log("pseudolabel accuracy/train", pl_acc)
        return sup_loss + self.alpha * cps_loss