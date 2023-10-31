import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.utils import plot_confusion_matrix
from .supervised import Supervised


class Multilabel(Supervised):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network = network(
            in_channels=in_channels,
            classes=num_classes - 1
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        pos_weight = torch.Tensor(class_weights[1:]).view(1, num_classes - 1, 1, 1)
        self.bce = bce(pos_weight=pos_weight)
        self.dice = dice_loss(mode="multilabel")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.val_accuracy = M.Accuracy(task='multilabel', num_classes=num_classes)
        self.val_cm = M.ConfusionMatrix(task="multilabel", num_classes=num_classes)
        self.train_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.network(x)

    def logits2probas(self, logits):
        probas = torch.sigmoid(logits)
        confs, preds = torch.max(probas, axis=1)
        nodata_proba = torch.unsqueeze(1 - confs, 1)
        all_probas = torch.cat([nodata_proba, probas], axis=1)
        return all_probas

    def probas2confpreds(self, probas):
        aux_confs, aux_preds = torch.max(probas[:, 1:, ...], axis=1)
        cond = aux_confs > 0.6
        preds = torch.where(cond, aux_preds + 1, 0)
        confs = torch.where(cond, aux_confs, 1 - aux_confs)
        return confs, preds
    
    def on_train_epoch_start(self):
        self.train_accuracy.reset()
        
    def on_train_epoch_end(self):
        self.log("accuracy/train", self.train_accuracy.compute())
    
    def one_hot(self, labels):
        one_hot = nn.functional.one_hot(labels, self.num_classes)
        return one_hot.permute(0,3,1,2)[:,1:,...].float()

    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        inputs = batch["image"]
        labels = batch["label"]
        onehot_labels = self.one_hot(labels)
        logits = self.network(inputs)  # B,C or C-1,H,W
        bce = self.bce(logits, onehot_labels)
        self.log(f"cross_entropy/train", bce)
        dice = self.dice(logits, onehot_labels)
        self.log(f"dice/train", dice)
        return self.ce_weight * bce + self.dice_weight * dice
    
    def on_validation_epoch_start(self):
        self.val_accuracy.reset()
        self.val_cm.reset()

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        onehot_labels = self.one_hot(labels)
        logits = self.forward(inputs)
        bce = self.bce(logits, onehot_labels)
        self.log(f"cross_entropy/val", bce)
        dice = self.dice(logits, onehot_labels)
        self.log(f"dice/val", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits))
        self.val_accuracy.update(preds, labels)
        self.val_cm.update(preds, labels)
    
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        confmat = self.val_cm.compute().detach().cpu()
        class_names = self.trainer.datamodule.class_names
        fig = plot_confusion_matrix(confmat, class_names, "recall", fontsize=8)
        logger = self.trainer.logger
        if logger:
            logger.experiment.add_figure("Recall matrix", fig, global_step=self.trainer.global_step)
    
    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        inputs = batch["image"]
        logits = self.forward(inputs)
        return logits