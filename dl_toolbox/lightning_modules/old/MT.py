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

from dl_toolbox.lightning_modules import BaseModule
import numpy as np
from dl_toolbox.augmentations import Cutmix,Cutmix2

class MT(BaseModule):

    # MT = Mean Teacher

    def __init__(self,
                 network,
                 weights,
                 ignore_index,
                 final_alpha,
                 alpha_milestones,
                 pseudo_threshold,
                 ema,
                 consist_aug,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        net_cls = self.net_factory.create(network)
        self.network1 = net_cls(*args, **kwargs)
        self.network2 = deepcopy(self.network1)
        self.num_classes = self.network1.out_channels
        out_dim = self.network1.out_dim
        self.weights = list(weights) if len(weights)>0 else [1]*self.num_classes
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            weight=torch.Tensor(self.weights)
        )
        self.mt_loss = nn.CrossEntropyLoss(
            reduction='none'
        )
        self.ema = ema
        self.final_alpha = final_alpha
        self.alpha_milestones = alpha_milestones
        self.pseudo_threshold = pseudo_threshold
        self.consist_aug = get_transforms(consist_aug)
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ema", type=float)
        parser.add_argument("--final_alpha", type=float)
        parser.add_argument("--alpha_milestones", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--consist_aug", type=str)
        parser.add_argument("--ignore_index", type=int)
        parser.add_argument("--network", type=str)
        parser.add_argument("--weights", type=float, nargs="+", default=())

        return parser

    def forward(self, x):
 
        return self.network1(x)

    def on_train_epoch_start(self):
        
        self.alpha = utils.ramp_down(
            self.trainer.current_epoch,
            *self.alpha_milestones,
            self.final_alpha
        )

    def update_teacher(self):

        # Update teacher model in place AFTER EACH BATCH?
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.network2.parameters(),
                                  self.network1.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

    def training_step(self, batch, batch_idx):

        batch, unsup_batch = batch["sup"], batch["unsup"]
        
        inputs = batch['image']
        labels = batch['mask']
        logits = self.network1(inputs)
        loss = self.loss(logits, labels)
        self.log('Train_sup_CE', loss)
        batch['logits'] = logits.detach()
        
        outputs = {'batch': batch, "loss": loss}

        if self.alpha > 0.:
            
            unsup_inputs = unsup_batch['image']

            with torch.no_grad():
                teacher_probs = self.network2(unsup_inputs).softmax(dim=1) # BxCxHxW

            cutmixed_inputs, cutmixed_probs = self.cutmix(
                input_batch=unsup_inputs,
                target_batch=teacher_probs
            ) # BxCxHxW
            unsup_batch['image'] = cutmixed_inputs
            cutmix_confs, cutmix_preds = torch.max(cutmixed_probs, dim=1) # BxHxW
            cutmixed_logits = self.network1(cutmixed_inputs)
            unsup_batch['logits'] = cutmixed_logits.detach()
            outputs['unsup_batch'] = unsup_batch
            loss_no_reduce = self.unsup_loss(
                cutmixed_logits,
                cutmix_preds
            ) # BxHxW

            cutmix_certain = cutmix_confs > self.pseudo_threshold # BxHxW
            certain = torch.sum(cutmix_certain)
            cutmix_loss = torch.sum(cutmix_certain * loss_no_reduce) / certain

            self.log('Cutmix consistency loss', cutmix_loss)

            loss += self.alpha * cutmix_loss

        self.update_teacher()
        self.log('Prop unsup train', self.alpha)
        self.log("Train_loss", loss)
        outputs['loss'] = loss

        return outputs
    
    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        labels = batch['mask']
        logits = outs['logits']
        loss = self.loss(logits, labels)
        self.log('Val_CE', loss)

        return outs    