from copy import deepcopy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.losses import DiceLoss, ProbOhemCrossEntropy2d
from dl_toolbox.utils import plot_confusion_matrix
from .supervised import Supervised


class MeanTeacher(Supervised):
    ### I tried to follow https://github.com/CuriousAI/mean-teacher/tree/master
    def __init__(
        self,
        alpha_ramp,
        teacher,
        consistency_loss,
        consistency_tf,
        ema_ramp,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.student = self.network
        self.teacher = teacher(
            in_channels=kwargs['in_channels'], num_classes=kwargs['num_classes']
        )
        self.alpha_ramp = alpha_ramp
        self.consistency_loss = consistency_loss
        self.consistency_tf = consistency_tf
        self.ema_ramp = ema_ramp

    def forward(self, x):
        return self.student(x)

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)
        self.ema = self.ema_ramp(self.trainer.current_epoch)
        self.log("Ema", self.ema)

    def training_step(self, batch, batch_idx):
        sup_loss = super().training_step(batch, batch_idx)
        unsup_inputs = batch["unsup"]["image"]
        unsup_inputs_1, _ = self.consistency_tf(unsup_inputs)
        unsup_inputs_2, _ = self.consistency_tf(unsup_inputs)
        with torch.no_grad():
            teacher_logits = self.teacher(unsup_inputs_1)
        student_logits = self.student(unsup_inputs_2)
        consistency = self.consistency_loss(student_logits, teacher_logits)
        self.log("consistency/train", consistency)
        return sup_loss + self.alpha * consistency

    def on_train_batch_end(self, outputs, batch, batch_idx):
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)