import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as M
import matplotlib.pyplot as plt

from dl_toolbox.losses import DiceLoss, ProbOhemCrossEntropy2d
from dl_toolbox.utils import plot_confusion_matrix
from torch.nn.functional import one_hot


class Mixmatch(pl.LightningModule):
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
        mix_tf,
        temperature,
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
        self.mix_tf = mix_tf
        self.temp = temperature
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
        xs_weak, ys = self.weak_tf(xs, ys)
        logits_xs_weak = self.network(xs_weak)
        ce = self.ce(logits_xs_weak, ys)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits_xs_weak, ys)
        self.log(f"dice/train", dice)
        # Mixmatch    
        ys = one_hot(ys, self.num_classes).float()
        if len(ys.shape) > 2: ys = ys.permute(0,3,1,2)
        xu = unsup_batch["image"]
        xu_weaks = [self.weak_tf(xu, None)[0] for _ in range(4)]
        xu_weak = torch.vstack(xu_weaks)
        with torch.no_grad():
            logits_xu_weak = self.network(xu_weak)
            chunks = torch.stack(torch.chunk(logits_xu_weak, chunks=4))
            probs_xu_weak = self.logits2probas(chunks.sum(dim=0))
            probs_xu_weak = probs_xu_weak ** (1.0 / self.temp)
            yu_sharp = probs_xu_weak / probs_xu_weak.sum(dim=-1, keepdim=True)
        yu = yu_sharp.repeat([4] + [1] * (len(yu_sharp.shape) - 1)) 
        x_weak = torch.vstack((xs_weak, xu_weak))
        y = torch.vstack((ys, yu))
        idxs = torch.randperm(y.shape[0])
        x_perm, y_perm = x_weak[idxs], y[idxs]
        b_s = ys.shape[0]
        xs_mix, ys_mix = self.mix_tf(xs_weak, ys, x_perm[:b_s], y_perm[:b_s])
        xu_mix, yu_mix = self.mix_tf(xu_weak, yu, x_perm[b_s:], y_perm[b_s:])
        logits_xs_mix = self.network(xs_mix)
        logits_xu_mix = self.network(xu_mix)
        ce_s = self.ce(logits_xs_mix, ys_mix)
        self.log("mixmatch ce sup/train", ce_s)
        ce_u = self.ce(logits_xu_mix, yu_mix)
        self.log("mixmatch ce unsup/train", ce_u)
        # Summing losses
        return self.ce_weight * ce + self.dice_weight * dice + self.alpha * (ce_s + ce_u)
    
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
