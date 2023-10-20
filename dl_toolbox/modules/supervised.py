import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as M
import matplotlib.pyplot as plt
from dl_toolbox.utils import plot_confusion_matrix
from pytorch_lightning.utilities import rank_zero_info
import torch.nn as nn
import math

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
        batch_tf,
        tta,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network = network(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.ce = ce_loss
        self.dice = dice_loss
        self.batch_tf = batch_tf
        self.tta = tta
        metric_args = {'task':'multiclass', 'num_classes':num_classes}#, 'ignore_index':self.ce.ignore_index}
        self.train_accuracy = M.Accuracy(**metric_args)
        self.val_accuracy = M.Accuracy(**metric_args)
        self.test_accuracy = M.Accuracy(**metric_args)
        self.val_cm = M.ConfusionMatrix(**metric_args, normalize='true')
        self.test_cm = M.ConfusionMatrix(**metric_args, normalize='true')
        self.val_jaccard = M.JaccardIndex(**metric_args)
        self.test_jaccard = M.JaccardIndex(**metric_args)
        self.val_ece = M.CalibrationError(**metric_args)
        self.test_ece = M.CalibrationError(**metric_args)
        self.val_mce = M.CalibrationError(**metric_args, norm='max')
        self.test_mce = M.CalibrationError(**metric_args, norm='max')
    
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {sum([int(torch.numel(p)) for p in trainable_parameters])} "
            f"trainable parameters out of {sum([int(torch.numel(p)) for p in parameters])}."
        )
        optimizer = self.optimizer(params=trainable_parameters)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler(optimizer=optimizer),
                "interval": "epoch"
            },
        }

    def forward(self, x):
        return self.network.forward(x)

    def logits2probas(cls, logits):
        return logits.softmax(dim=1)

    def probas2confpreds(cls, probas):
        return torch.max(probas, dim=1)   
        
    def one_hot(self, y):
        return F.one_hot(y.unsqueeze(1), self.num_classes).transpose(1,-1).squeeze(-1).float()
    
    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        xs = batch["image"]
        ys = batch["label"]
        ys_o = self.one_hot(ys)
        xs, ys_o = self.batch_tf(xs, ys_o)
        logits_xs = self.forward(xs)
        ce = self.ce(logits_xs, ys_o)
        self.log(f"cross_entropy/train", ce)
        dice = 0
        if self.dice is not None: 
            dice = self.dice(logits_xs, ys_o)
            self.log(f"dice/train", dice)
        return ce + dice
        
    def validation_step(self, batch, batch_idx):
        xs = batch["image"]
        ys = batch["label"]
        ys_o = self.one_hot(ys)
        logits_xs = self.forward(xs)
        ce = self.ce(logits_xs, ys_o)
        self.log(f"cross_entropy/val", ce)
        if self.dice is not None: 
            dice = self.dice(logits_xs, ys_o)
            self.log(f"dice/val", dice)
        probs = self.logits2probas(logits_xs)
        _, preds = self.probas2confpreds(probs)
        self.val_accuracy.update(preds, ys)
        self.val_cm.update(preds, ys)
        self.val_jaccard.update(preds, ys)
        self.val_ece.update(probs, ys)
        self.val_mce.update(probs, ys)
        
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        self.log("iou/val", self.val_jaccard.compute())
        self.log("ece/val", self.val_ece.compute())
        self.log("mce/val", self.val_mce.compute())
        confmat = self.val_cm.compute().detach().cpu()
        self.val_accuracy.reset()
        self.val_jaccard.reset()
        self.val_cm.reset()
        self.val_ece.reset()
        self.val_mce.reset()
        class_names = self.trainer.datamodule.class_names
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        try:
            logger.experiment.add_figure("confmat/val", fig, global_step=self.trainer.global_step)
        except:
            pass


    def test_step(self, batch, batch_idx):
        xs = batch["image"]
        ys = batch["label"]
        ys_o = self.one_hot(ys)
        logits_xs = self.forward(xs)
        ce = self.ce(logits_xs, ys_o)
        self.log(f"cross_entropy/test", ce)
        if self.dice is not None: 
            dice = self.dice(logits_xs, ys_o)
            self.log(f"dice/test", dice)
        if self.tta is not None:
            auxs = [self.forward(x) for x in self.tta(xs)]
            logits_xs = torch.stack([logits_xs] + self.tta.revert(xs, auxs)).sum(dim=0)
        probs = self.logits2probas(logits_xs)
        _, preds = self.probas2confpreds(probs)
        self.test_accuracy.update(preds, ys)
        self.test_jaccard.update(preds, ys)
        self.test_cm.update(preds, ys)
        self.test_ece.update(probs, ys)
        self.test_mce.update(probs, ys)
        
    def on_test_epoch_end(self):
        self.log("accuracy/test", self.test_accuracy.compute())
        self.log("iou/test", self.test_jaccard.compute())
        self.log("ece/test", self.test_ece.compute())
        self.log("mce/test", self.test_mce.compute())
        confmat = self.test_cm.compute().detach().cpu()
        self.test_accuracy.reset()
        self.test_jaccard.reset()
        self.test_cm.reset()
        self.test_ece.reset()
        self.test_mce.reset()
        class_names = self.trainer.datamodule.class_names
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        try:
            logger.experiment.add_figure("confmat/test", fig, global_step=self.trainer.global_step)
        except:
            pass

    def predict_step(self, batch, batch_idx):
        xs = batch["image"]
        logits_xs = self.forward(xs)
        if self.tta is not None:
            auxs = [self.forward(x) for x in self.tta(xs)]
            logits_xs = torch.stack([logits_xs] + self.tta.revert(xs, auxs)).sum(dim=0)
        return logits_xs


#class Supervised(pl.LightningModule):
#    def __init__(
#        self,
#        network,
#        in_channels,
#        num_classes,
#        optimizer,
#        scheduler,
#        ce_loss,
#        dice_loss,
#        batch_tf,
#        tta,
#        *args,
#        **kwargs
#    ):
#        super().__init__()
#        self.network = network(in_channels=in_channels, num_classes=num_classes)
#        self.optimizer = optimizer
#        self.scheduler = scheduler
#        self.num_classes = num_classes
#        self.ce = ce_loss
#        self.dice = dice_loss
#        self.batch_tf = batch_tf
#        self.tta = tta
#        metric_args = {'task':'multiclass', 'num_classes':num_classes}#, 'ignore_index':self.ce.ignore_index}
#        self.train_accuracy = M.Accuracy(**metric_args)
#        self.val_accuracy = M.Accuracy(**metric_args)
#        self.test_accuracy = M.Accuracy(**metric_args)
#        self.val_cm = M.ConfusionMatrix(**metric_args, normalize='true')
#        self.test_cm = M.ConfusionMatrix(**metric_args, normalize='true')
#        self.val_jaccard = M.JaccardIndex(**metric_args)
#        self.test_jaccard = M.JaccardIndex(**metric_args)
#        self.val_ece = M.CalibrationError(**metric_args)
#        self.test_ece = M.CalibrationError(**metric_args)
#        self.val_mce = M.CalibrationError(**metric_args, norm='max')
#        self.test_mce = M.CalibrationError(**metric_args, norm='max')
#
#    def configure_optimizers(self):
#        optimizer = self.optimizer(params=self.parameters())
#        return {
#            "optimizer": optimizer,
#            "lr_scheduler": {
#                "scheduler": self.scheduler(optimizer=optimizer),
#                "interval": "epoch"
#            },
#        }
#
#    def forward(self, x):
#        return self.network(x)
#
#    def logits2probas(cls, logits):
#        return logits.softmax(dim=1)
#
#    def probas2confpreds(cls, probas):
#        return torch.max(probas, dim=1)   
#        
#    def one_hot(self, y):
#        return F.one_hot(y.unsqueeze(1), self.num_classes).transpose(1,-1).squeeze(-1).float()
#    
#    def training_step(self, batch, batch_idx):
#        batch = batch["sup"]
#        xs = batch["image"]
#        ys = batch["label"]
#        ys_o = self.one_hot(ys)
#        xs, ys_o = self.batch_tf(xs, ys_o)
#        logits_xs = self.network(xs)
#        ce = self.ce(logits_xs, ys_o)
#        self.log(f"cross_entropy/train", ce)
#        dice = 0
#        if self.dice is not None: 
#            dice = self.dice(logits_xs, ys_o)
#            self.log(f"dice/train", dice)
#        return ce + dice
#        
#    def validation_step(self, batch, batch_idx):
#        xs = batch["image"]
#        ys = batch["label"]
#        ys_o = self.one_hot(ys)
#        logits_xs = self.forward(xs)
#        ce = self.ce(logits_xs, ys_o)
#        self.log(f"cross_entropy/val", ce)
#        if self.dice is not None: 
#            dice = self.dice(logits_xs, ys_o)
#            self.log(f"dice/val", dice)
#        probs = self.logits2probas(logits_xs)
#        _, preds = self.probas2confpreds(probs)
#        self.val_accuracy.update(preds, ys)
#        self.val_cm.update(preds, ys)
#        self.val_jaccard.update(preds, ys)
#        self.val_ece.update(probs, ys)
#        self.val_mce.update(probs, ys)
#        
#    def on_validation_epoch_end(self):
#        self.log("accuracy/val", self.val_accuracy.compute())
#        self.log("iou/val", self.val_jaccard.compute())
#        self.log("ece/val", self.val_ece.compute())
#        self.log("mce/val", self.val_mce.compute())
#        confmat = self.val_cm.compute().detach().cpu()
#        self.val_accuracy.reset()
#        self.val_jaccard.reset()
#        self.val_cm.reset()
#        self.val_ece.reset()
#        self.val_mce.reset()
#        class_names = self.trainer.datamodule.class_names
#        logger = self.trainer.logger
#        fs = 12 - 2*(self.num_classes//10)
#        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
#        try:
#            logger.experiment.add_figure("confmat/val", fig, global_step=self.trainer.global_step)
#        except:
#            pass
#
#
#    def test_step(self, batch, batch_idx):
#        xs = batch["image"]
#        ys = batch["label"]
#        ys_o = self.one_hot(ys)
#        logits_xs = self.forward(xs)
#        ce = self.ce(logits_xs, ys_o)
#        self.log(f"cross_entropy/test", ce)
#        if self.dice is not None: 
#            dice = self.dice(logits_xs, ys_o)
#            self.log(f"dice/test", dice)
#        if self.tta is not None:
#            auxs = [self.forward(x) for x in self.tta(xs)]
#            logits_xs = torch.stack([logits_xs] + self.tta.revert(xs, auxs)).sum(dim=0)
#        probs = self.logits2probas(logits_xs)
#        _, preds = self.probas2confpreds(probs)
#        self.test_accuracy.update(preds, ys)
#        self.test_jaccard.update(preds, ys)
#        self.test_cm.update(preds, ys)
#        self.test_ece.update(probs, ys)
#        self.test_mce.update(probs, ys)
#        
#    def on_test_epoch_end(self):
#        self.log("accuracy/test", self.test_accuracy.compute())
#        self.log("iou/test", self.test_jaccard.compute())
#        self.log("ece/test", self.test_ece.compute())
#        self.log("mce/test", self.test_mce.compute())
#        confmat = self.test_cm.compute().detach().cpu()
#        self.test_accuracy.reset()
#        self.test_jaccard.reset()
#        self.test_cm.reset()
#        self.test_ece.reset()
#        self.test_mce.reset()
#        class_names = self.trainer.datamodule.class_names
#        logger = self.trainer.logger
#        fs = 12 - 2*(self.num_classes//10)
#        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
#        try:
#            logger.experiment.add_figure("confmat/test", fig, global_step=self.trainer.global_step)
#        except:
#            pass
#
#    def predict_step(self, batch, batch_idx):
#        xs = batch["image"]
#        logits_xs = self.forward(xs)
        return logits_xs