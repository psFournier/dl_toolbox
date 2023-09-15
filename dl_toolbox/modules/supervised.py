import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt
from dl_toolbox.utils import plot_confusion_matrix


class Supervised(pl.LightningModule):
    def __init__(
        self,
        network,
        in_channels,
        num_classes,
        optimizer,
        scheduler,
        ce_loss,
        dice_loss,
        class_weights,
        ce_weight,
        dice_weight,
        tf,
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
        self.tf = tf
        self.train_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_cm = M.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_cm = M.ConfusionMatrix(task="multiclass", num_classes=num_classes)

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
        
    def on_train_epoch_end(self):
        self.log("accuracy/train", self.train_accuracy.compute())
        self.train_accuracy.reset()
    
    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        xs = batch["image"]
        ys = batch["label"]
        xs, ys = self.tf(xs, ys)
        logits_xs = self.network(xs)
        ce = self.ce(logits_xs, ys)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits_xs, ys)
        self.log(f"dice/train", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits_xs))
        self.train_accuracy.update(preds, ys)
        return self.ce_weight * ce + self.dice_weight * dice
        
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
        
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        confmat = self.val_cm.compute().detach().cpu()
        class_names = self.trainer.datamodule.class_names
        fig = plot_confusion_matrix(confmat, class_names, "recall", fontsize=8)
        logger = self.trainer.logger
        if logger:
            logger.experiment.add_figure("Recall matrix", fig, global_step=self.trainer.global_step)
        self.val_accuracy.reset()
        self.val_cm.reset()

    def test_step(self, batch, batch_idx):
        xs = batch['image']
        ys = batch["label"]
        logits = self.forward(xs)
        ce = self.ce(logits, ys)
        self.log(f"cross_entropy/test", ce)
        dice = self.dice(logits, ys)
        self.log(f"dice/test", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits))
        self.test_accuracy.update(preds, ys)
        self.test_cm.update(preds, ys)
        
    def on_test_epoch_end(self):
        self.log("accuracy/test", self.test_accuracy.compute())
        confmat = self.test_cm.compute().detach().cpu()
        self.test_accuracy.reset()
        self.test_cm.reset()
        class_names = self.trainer.datamodule.class_names
        fig = plot_confusion_matrix(confmat, class_names, "recall", fontsize=8)
        logger = self.trainer.logger
        if logger:
            logger.experiment.add_figure("Recall matrix", fig, global_step=self.trainer.global_step)
        else:
            plt.savefig('/tmp/last_conf_mat.png')

    def predict_step(self, batch, batch_idx):
        inputs = batch["image"]
        logits = self.forward(inputs)
        return logits