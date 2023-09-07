from copy import deepcopy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.losses import DiceLoss, ProbOhemCrossEntropy2d
from dl_toolbox.utils import plot_confusion_matrix


class MeanTeacher(pl.LightningModule):
    ### I tried to follow https://github.com/CuriousAI/mean-teacher/tree/master
    def __init__(
        self,
        student,
        teacher,
        optimizer,
        scheduler,
        ce_loss,
        dice_loss,
        ce_weight,
        dice_weight,
        alpha_ramp,
        class_weights,
        in_channels,
        num_classes,
        consistency_loss,
        consistency_aug,
        ema_ramp,
        *args,
        **kwargs
    ):
        super().__init__()
        self.student = student(in_channels=in_channels, num_classes=num_classes)
        self.teacher = teacher(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.ce = ce_loss(weight=torch.Tensor(class_weights))
        self.dice = dice_loss(mode="multiclass")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.alpha_ramp = alpha_ramp
        self.val_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_cm = M.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.train_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)
        self.consistency_loss = consistency_loss
        self.consistency_aug = consistency_aug
        self.ema_ramp = ema_ramp

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.student(x)

    def logits2probas(cls, logits):
        return logits.softmax(dim=1)

    def probas2confpreds(cls, probas):
        return torch.max(probas, dim=1)
    
    def on_train_epoch_start(self):
        self.train_accuracy.reset()
        
    def on_train_epoch_end(self):
        self.log("accuracy/train", self.train_accuracy.compute())

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)
        self.ema = self.ema_ramp(self.trainer.current_epoch)
        self.log("Ema", self.ema)

    def training_step(self, batch, batch_idx):
        batch, unsup_batch = batch["sup"], batch["unsup"]
        xs = batch["image"]
        ys = batch["label"]
        logits_xs = self.student(xs)
        ce = self.ce(logits_xs, ys)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits_xs, ys)
        self.log(f"dice/train", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits_xs))
        self.train_accuracy.update(preds, ys)
        unsup_inputs = unsup_batch["image"]
        unsup_inputs_1, _ = self.consistency_aug(unsup_inputs)
        unsup_inputs_2, _ = self.consistency_aug(unsup_inputs)
        with torch.no_grad():
            teacher_logits = self.teacher(unsup_inputs_1)
        student_logits = self.student(unsup_inputs_2)
        consistency_loss = self.consistency_loss(student_logits, teacher_logits)
        self.log("Consistency loss", consistency_loss)
        return self.ce_weight * ce + self.dice_weight * dice + self.alpha * consistency_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)
            
    def on_validation_epoch_start(self):
        self.val_accuracy.reset()
        self.val_cm.reset()
        
    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        logits = self.forward(inputs)
        ce = self.ce(logits, labels)
        self.log(f"cross_entropy/val", ce)
        dice = self.dice(logits, labels)
        self.log(f"dice/val", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits))
        self.val_accuracy.update(preds, labels)
        self.val_cm.update(preds, labels)
        # Checking pseudo_labels quality
        teacher_logits = self.teacher(inputs)
        teacher_probas = self.logits2probas(teacher_logits)
        teacher_confs, teacher_preds = self.probas2confpreds(teacher_probas)
        teacher_accus = teacher_preds.eq(labels).float().mean()
        self.log("Val acc of teacher labels", teacher_accus)
        
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        confmat = self.val_cm.compute().detach().cpu()
        class_names = self.trainer.datamodule.class_names
        fig = plot_confusion_matrix(confmat, class_names, "recall", fontsize=8)
        logger = self.trainer.logger
        if logger:
            logger.experiment.add_figure("Recall matrix", fig, global_step=self.trainer.global_step)