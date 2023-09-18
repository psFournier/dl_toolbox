import pytorch_lightning as pl
import torch
import torch.nn as nn
from .supervised import Supervised


class Fixmatch(Supervised):
    def __init__(
        self,
        alpha_ramp,
        weak_tf,
        strong_tf,
        ce_loss_u,
        threshold,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha_ramp = alpha_ramp
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf # Weak and strong aug must not touch labels
        self.ce_u = ce_loss_u()
        self.threshold = threshold
        
    def forward(self, x):
        return self.network(x)

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)

    def training_step(self, batch, batch_idx):
        batch, unsup_batch = batch["sup"], batch["unsup"]
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