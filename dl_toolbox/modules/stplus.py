import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.losses import DiceLoss, ProbOhemCrossEntropy2d
from dl_toolbox.utils import plot_confusion_matrix


class Fixmatch(pl.LightningModule):
    def __init__(
        self,
        network,
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
        weak_tf,
        strong_tf,
        ce_loss_u,
        threshold,
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
        self.alpha_ramp = alpha_ramp
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf
        self.ce_u = ce_loss_u()
        self.threshold = threshold
        self.val_accuracy = M.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_cm = M.ConfusionMatrix(task="multiclass", num_classes=num_classes)

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
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)

    def training_step(self, batch, batch_idx):
        batch, unsup_batch = batch["sup"], batch["unsup"]
        xs = batch["image"]
        ys = batch["label"]
        xs_weak, ys_weak = self.weak_tf(xs, ys)
        logits_xs_weak = self.network(xs_weak)
        ce = self.ce(logits_xs_weak, ys_weak)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits_xs_weak, ys_weak)
        self.log(f"dice/train", dice)
        # Fixmatch part    
        xu = unsup_batch["image"]
        xu_weak, _ = self.weak_tf(xu, None)
        xu_strong, _ = self.strong_tf(xu, None)
        with torch.no_grad():
            logits_xu_weak = self.network(xu_weak)
            probs_xu_weak = self.logits2probas(logits_xu_weak)
            conf_xu_weak, pl_xu_weak = self.probas2confpreds(probs_xu_weak)
            certain_xu_weak = conf_xu_weak.ge(self.threshold).float()
        logits_xu_strong = self.network(xu_strong)
        xu_ce = self.ce_u(logits_xu_strong, pl_xu_weak)
        certain_xu_weak_sum = torch.sum(certain_xu_weak) + 1e-5
        xu_ce_mean = torch.sum(certain_xu_weak * xu_ce) / certain_xu_weak_sum
        self.log("fixmatch ce/train", xu_ce_mean)
        # Summing losses
        return self.ce_weight * ce + self.dice_weight * dice + self.alpha * xu_ce_mean
    
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
