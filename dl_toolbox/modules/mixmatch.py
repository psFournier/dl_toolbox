import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.losses import DiceLoss, ProbOhemCrossEntropy2d
from dl_toolbox.utils import plot_confusion_matrix


class CrossPseudoSupervision(pl.LightningModule):
    def __init__(
        self,
        network1,
        network2,
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
        *args,
        **kwargs
    ):
        super().__init__()
        self.network1 = network1(in_channels=in_channels, num_classes=num_classes)
        self.network2 = network2(in_channels=in_channels, num_classes=num_classes)
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

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.network1(x)

    def logits2probas(cls, logits):
        return logits.softmax(dim=1)

    def probas2confpreds(cls, probas):
        return torch.max(probas, dim=1)

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)

    def training_step(self, batch, batch_idx):
        batch, unsup_batch = batch["sup"], batch["unsup"]
        inputs = batch["image"]
        labels = batch["label"]
        logits1 = self.network1(inputs)
        logits2 = self.network2(inputs)
        ce = self.ce(logits1, labels) + self.ce(logits2, labels)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits1, labels)+self.dice(logits2, labels)
        self.log(f"dice/train", dice)
            
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
        
        pl_acc = sup_pl_1.eq(labels).float().mean()
        self.log("pseudolabel accuracy/train", pl_acc)
    
        return self.ce_weight * ce + self.dice_weight * dice + self.alpha * (cps_loss_labeled + cps_loss_unlabeled)
    
    def on_validation_epoch_start(self):
        self.val_accuracy.reset()
        self.val_cm.reset()
        
    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        logits1 = self.network1(inputs)
        logits2 = self.network2(inputs)
        logits = (logits1 + logits2) / 2
        ce = (self.ce(logits1, labels) + self.ce(logits2, labels)) / 2
        self.log(f"cross_entropy/val", ce)
        dice = (self.dice(logits1, labels)+self.dice(logits2, labels))/2
        self.log(f"dice/val", dice)
        _, pl_1 = torch.max(logits1, dim=1)
        self.log("cps_loss_labeled/val", self.ce(logits2, pl_1))
        pl_acc = pl_1.eq(labels).float().mean()
        self.log("pseudolabel accuracy/val", pl_acc)
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

    def predict_step(self, batch, batch_idx):
        inputs = batch["image"]
        logits = self.forward(inputs)
        return logits
    
    
            ## Supervising network 2 with pseudolabels from network 1
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
