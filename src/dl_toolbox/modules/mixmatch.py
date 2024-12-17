import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.nn.functional import one_hot
from .supervised import Supervised


class Mixmatch(Supervised):
    def __init__(
        self,
        alpha_ramp,
        weak_tf,
        mix_tf,
        temperature,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha_ramp = alpha_ramp
        self.weak_tf = weak_tf # Must not touch labels
        self.mix_tf = mix_tf
        self.temp = temperature

    def forward(self, x):
        return self.network.forward(self.norm(x))

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)

    def training_step(self, batch, batch_idx):
        sup_loss = super().training_step(batch, batch_idx)      
        # Mixmatch
        xu = batch["unsup"]["image"]
        xu_weaks = [self.weak_tf(xu, None)[0] for _ in range(2)]
        xu_weak = torch.vstack(xu_weaks)
        with torch.no_grad():
            logits_xu_weak = self.forward(xu_weak)
            chunks = torch.stack(torch.chunk(logits_xu_weak, chunks=2))
            probs_xu_weak = self.logits2probas(chunks.sum(dim=0))
            probs_xu_weak = probs_xu_weak ** (1.0 / self.temp)
            yu_sharp = probs_xu_weak / probs_xu_weak.sum(dim=1, keepdim=True)
        yu = yu_sharp.repeat([2] + [1] * (len(yu_sharp.shape) - 1)) 
    
        xs = batch["sup"]["image"]
        ys = batch["sup"]["label"]
        x_weak = torch.vstack((xs, xu_weak))
        ys_o = self.one_hot(ys)
        y = torch.vstack((ys_o, yu))
        
        idxs = torch.randperm(y.shape[0])
        x_perm, y_perm = x_weak[idxs], y[idxs]
        
        b_s = ys.shape[0]
        xs_mix, ys_mix = self.mix_tf(xs, ys_o, x_perm[:b_s], y_perm[:b_s])
        xu_mix, yu_mix = self.mix_tf(xu_weak, yu, x_perm[b_s:], y_perm[b_s:])
        logits_xs_mix = self.forward(xs_mix)
        logits_xu_mix = self.forward(xu_mix)
        ce_s = self.ce(logits_xs_mix, ys_mix)
        self.log("mixmatch ce sup/train", ce_s)
        ce_u = self.ce(logits_xu_mix, yu_mix)
        self.log("mixmatch ce unsup/train", ce_u)
        
        return sup_loss + self.alpha * (ce_s+ce_u)