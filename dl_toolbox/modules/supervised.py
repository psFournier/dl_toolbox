import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as M
import matplotlib.pyplot as plt
from dl_toolbox.utils import plot_confusion_matrix
from pytorch_lightning.utilities import rank_zero_info
import torch.nn as nn
import math
from dl_toolbox.transforms import Mixup

class Supervised(pl.LightningModule):
    def __init__(
        self,
        network,
        in_channels,
        num_classes,
        optimizer,
        scheduler,
        loss,
        batch_tf,
        metric_ignore_index,
        norm,
        one_hot,
        tta=None,
        sliding=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network = network(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.loss = loss
        self.batch_tf = batch_tf
        self.tta = tta
        self.sliding = sliding
        self.norm = norm
        self.one_hot = one_hot
        metric_args = {'task':'multiclass', 'num_classes':num_classes, 'ignore_index':metric_ignore_index}
        self.train_accuracy = M.Accuracy(**metric_args)
        self.val_accuracy = M.Accuracy(**metric_args)
        self.test_accuracy = M.Accuracy(**metric_args)
        self.val_cm = M.ConfusionMatrix(**metric_args, normalize='true')
        self.test_cm = M.ConfusionMatrix(**metric_args, normalize='true')
        self.val_jaccard = M.JaccardIndex(**metric_args)
        self.test_jaccard = M.JaccardIndex(**metric_args)
    
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {sum([int(torch.numel(p)) for p in trainable_parameters])} "
            f"trainable parameters out of {sum([int(torch.numel(p)) for p in parameters])}."
        )
        optimizer = self.optimizer(params=trainable_parameters)
        scheduler = self.scheduler(optimizer)
        #schs = self.scheduler.schedulers.values()
        #scheduler = torch.optim.lr_scheduler.SequentialLR(
        #    optimizer,
        #    schedulers=[s(optimizer) for s in schs],
        #    milestones=self.scheduler.milestones
        #)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }

    def forward(self, x, sliding=None, tta=None):
        if sliding is not None:
            auxs = [self.forward(aux, tta=tta) for aux in sliding(x)]
            return sliding.merge(auxs)
        elif tta is not None:
            auxs = [self.forward(aux) for aux in tta(x)]
            logits = self.forward(x)
            return torch.stack([logits] + self.tta.revert(auxs)).sum(dim=0)
        else:
            return self.network.forward(self.norm(x))
    
    def to_one_hot(self, y):
        return torch.movedim(F.one_hot(y, self.num_classes),-1,1).float()
    
    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        x = batch["image"]
        y = batch["label"]
        if self.one_hot: y = self.to_one_hot(y)
        x, y = self.batch_tf(x, y)
        logits_x = self.forward(x, sliding=None, tta=None)
        loss = self.loss(logits_x, y)
        self.log(f"{self.loss.__name__}/train", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        if self.one_hot: y = self.to_one_hot(y)
        logits_x = self.forward(x, sliding=self.sliding)                    
        loss = self.loss(logits_x, y)
        self.log(f"{self.loss.__name__}/val", loss)
        probs = self.loss.prob(logits_x)
        preds = self.loss.pred(probs)
        self.val_accuracy.update(preds, y)
        self.val_cm.update(preds, y)
        self.val_jaccard.update(preds, y)
        
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        self.log("iou/val", self.val_jaccard.compute())
        confmat = self.val_cm.compute().detach().cpu()
        self.val_accuracy.reset()
        self.val_jaccard.reset()
        self.val_cm.reset()
        class_names = self.trainer.datamodule.class_names
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        logger.experiment.add_figure("confmat/val", fig, global_step=self.trainer.global_step)

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        if self.one_hot: y = self.to_one_hot(y)
        logits_x = self.forward(x, sliding=self.sliding, tta=self.tta)
        loss = self.loss(logits_x, y)
        self.log(f"{self.loss.__name__}/test", loss)
        probs = self.loss.prob(logits_x)
        preds = self.loss.pred(probs)
        self.test_accuracy.update(preds, y)
        self.test_cm.update(preds, y)
        self.test_jaccard.update(preds, y)
        
    def on_test_epoch_end(self):
        self.log("accuracy/test", self.test_accuracy.compute())
        self.log("iou/test", self.test_jaccard.compute())
        confmat = self.test_cm.compute().detach().cpu()
        self.test_accuracy.reset()
        self.test_jaccard.reset()
        self.test_cm.reset()
        class_names = self.trainer.datamodule.class_names
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        logger.experiment.add_figure("confmat/test", fig, global_step=self.trainer.global_step)

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        logits_x = self.forward(x, sliding=self.sliding, tta=self.tta)
        return self.loss.prob(logits_x)