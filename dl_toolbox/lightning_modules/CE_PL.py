from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import torch
import torchmetrics.functional as torchmetrics
from dl_toolbox.losses import DiceLoss
from copy import deepcopy
import torch.nn.functional as F

from dl_toolbox.lightning_modules.utils import *
from dl_toolbox.lightning_modules import CE
from dl_toolbox.callbacks import plot_confusion_matrix, plot_calib, compute_calibration_bins, compute_conf_mat, log_batch_images
from dl_toolbox.torch_datasets.utils import *

class CE_PL(CE):

    def __init__(
        self,
        final_alpha,
        alpha_milestones,
        pseudo_threshold,
        unsup_aug,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.final_alpha = final_alpha
        self.alpha_milestones = alpha_milestones
        self.pseudo_threshold = pseudo_threshold
        self.unsup_aug = get_transforms(unsup_aug)
        self.alpha = 0.
        self.pl_loss = nn.CrossEntropyLoss(
            reduction='none'
        )
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--final_alpha", type=float)
        parser.add_argument("--alpha_milestones", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--unsup_aug", type=str)

        return parser

    def on_train_epoch_start(self):

        start = self.alpha_milestones[0]
        end = self.alpha_milestones[1]
        e = self.trainer.current_epoch
        alpha = self.final_alpha

        if e <= start:
            self.alpha = 0.
        elif e <= end:
            self.alpha = ((e - start) / (end - start)) * alpha
        else:
            self.alpha = alpha

    def training_step(self, batch, batch_idx):

        batch, unsup_batch = batch["sup"], batch["unsup"]
        self.log('Prop unsup train', self.alpha)
        outs = super().training_step(batch, batch_idx)
        
        if self.trainer.current_epoch >= self.alpha_milestones[0]:

            unsup_inputs = unsup_batch['image']
            pl_logits = self.network(unsup_inputs).detach()
            pl_probas = self._compute_probas(pl_logits)
            aug_unsup_inputs, aug_pl_probas = self.unsup_aug(
                img=unsup_inputs,
                label=pl_probas
            )
            pl_confs, pl_preds = self._compute_conf_preds(aug_pl_probas)
            pl_certain = (pl_confs > self.pseudo_threshold).float()
            pl_certain_sum = torch.sum(pl_certain) + 1e-5
            self.log('PL certainty prop', torch.mean(pl_certain))
            aug_pl_logits = self.network(aug_unsup_inputs)
            pl_loss = self.pl_loss(aug_pl_logits, pl_preds)
            pl_loss = torch.sum(pl_certain * pl_loss) / pl_certain_sum
            self.log('PL loss', pl_loss)
            outs['loss'] += self.alpha * pl_loss
            if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
                log_batch_images(
                    unsup_batch,
                    self.trainer,
                    prefix='Unsup_train'
                )

        return outs

    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        logits = outs['logits']
        probas = self._compute_probas(logits.detach())
        confs, preds = self._compute_conf_preds(probas)
        certain = (confs > self.pseudo_threshold).float()
        certain_sum = torch.sum(certain) + 1e-5
        self.log('Val certainty prop', torch.mean(certain))
        acc = preds.eq(batch['mask']).float()
        pl_acc = torch.sum(certain * acc) / certain_sum
        self.log('Val acc of pseudo labels', pl_acc)

        return outs


