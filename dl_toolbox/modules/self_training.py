import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.losses import DiceLoss, ProbOhemCrossEntropy2d
from dl_toolbox.utils import plot_confusion_matrix


class SelfTraining(pl.LightningModule):
    def __init__(
        self,
        network,
        optimizer,
        scheduler,
        ce_loss,
        dice_loss,
        ce_weight,
        dice_weight,
        class_weights,
        in_channels,
        num_classes,
        weak_tf,
        strong_tf,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network = network(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.ce = ce_loss(weight=torch.Tensor(class_weights))
        self.dice = dice_loss(mode="multiclass")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf
        self.val_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_cm = M.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.train_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.network(x)

    def logits2probas(cls, logits):
        return logits.softmax(dim=1)

    def probas2confpreds(cls, probas):
        return torch.max(probas, dim=1)
    
    def on_train_epoch_start(self):
        self.train_accuracy.reset()
        
    def on_train_epoch_end(self):
        self.log("accuracy/train", self.train_accuracy.compute())

    def training_step(self, batch, batch_idx):
        batch, pseudosup_batch = batch["sup"], batch["pseudosup"]
        xs = batch["image"]
        ys = batch["label"]
        xs_weak, ys_weak = self.weak_tf(xs, ys)
        # ST    
        x_pl = pseudosup_batch["image"]
        prob_y_pl = pseudosup_batch["label"]
        _, y_pl = torch.max(prob_y_pl, dim=1)
        x_pl_strong, y_pl_strong = self.strong_tf(x_pl, y_pl)
        x = torch.vstack((xs_weak, x_pl_strong))
        y = torch.vstack((ys_weak, y_pl_strong))
        logits_x = self.network(x)
        ce = self.ce(logits_x, y)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits_x, y)
        self.log(f"dice/train", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits_x))
        self.train_accuracy.update(preds, y)
        return self.ce_weight * ce + self.dice_weight * dice
    
    def on_validation_epoch_start(self):
        self.val_accuracy.reset()
        self.val_cm.reset()
        
    def validation_step(self, batch, batch_idx):
        xs = batch["image"]
        ys = batch["label"]
        logits = self.network(xs)
        ce = self.ce(logits, ys)
        self.log(f"cross_entropy/val", ce)
        dice = self.dice(logits, ys)
        self.log(f"dice/val", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits))
        self.val_accuracy.update(preds, ys)
        self.val_cm.update(preds, ys)
        
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        confmat = self.val_cm.compute().detach().cpu()
        class_names = self.trainer.datamodule.class_names
        fig = plot_confusion_matrix(confmat, class_names, "recall", fontsize=8)
        logger = self.trainer.logger
        if logger:
            logger.experiment.add_figure("Recall matrix", fig, global_step=self.trainer.global_step)

    def predict_step(self, batch, batch_idx):
        inputs = batch["image"]
        logits = self.forward(inputs)
        return logits
    
    
            ## Supervising network 2 with pseudoys from network 1
            # with torch.no_grad():
            #    pseudo_probs_1 = self.logits2probas(unsup_logits_1)
            # pseudo_confs_1, pseudo_preds_1 = self.probas2confpreds(pseudo_probs_1)
            # loss_no_reduce_2 = self.unsup_loss(
            #    unsup_logits_2,
            #    pseudo_preds_1
            # ) # B,H,W
            # pseudo_certain_1 = (pseudo_confs_1 > self.pseudo_threshold).float() # B,H,W
            # pseudo_certain_1_sum = torch.sum(pseudo_certain_1) + 1e-5
            # self.log('Pseudo_certain_1', torch.mean(pseudo_certain_1))
            # pseudo_loss_2 = torch.sum(pseudo_certain_1 * loss_no_reduce_2) / pseudo_certain_1_sum
