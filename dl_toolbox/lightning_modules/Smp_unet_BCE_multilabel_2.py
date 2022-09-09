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
from dl_toolbox.lightning_modules import BaseModule
from dl_toolbox.utils import TorchOneHot
from dl_toolbox.networks import *



class Smp_Unet_BCE_multilabel_2(BaseModule):

    # BCE_multilabel = Binary Cross Entropy for multilabel prediction

    def __init__(self,
                 encoder,
                 in_channels,
                 pretrained=True,
                 initial_lr=0.05,
                 final_lr=0.001,
                 lr_milestones=(0.5,0.9),
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        
        self.network = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=self.num_classes - 1,
            decoder_use_batchnorm=True
        )
        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)
        self.bce = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.Tensor(self.weights[1:]).reshape(1, -1, 1, 1)
        )
        self.onehot = TorchOneHot(range(self.num_classes))
        self.dice = DiceLoss(
            mode="multilabel",
            log_loss=False,
            from_logits=True
        )
        self.ignore_index = 0
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)

        return parser

    def forward(self, x):
        
        return self.network(x)

    def training_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        onehot_labels = self.onehot(labels).float() # B,C,H,W

        final_labels = onehot_labels[:, 1:, ...] # B,C-1,H,W
        
        mask = torch.ones_like(
            final_labels,
            dtype=onehot_labels.dtype,
            device=onehot_labels.device
        )
        
        logits = self.network(inputs) # B,C-1,H,W
        bce = self.bce(logits, final_labels)
        bce = torch.sum(mask * bce) / torch.sum(mask)
        dice = self.dice(logits * mask, final_labels * mask)
        loss = bce + dice
        
        self.log('Train_sup_BCE', bce)
        self.log('Train_sup_Dice', dice)
        self.log('Train_sup_loss', loss)

        batch['logits'] = logits.detach()

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.forward(inputs)
        probas = torch.sigmoid(logits) # B,C-1,H,W
        confidences, preds = torch.max(probas, dim=1)
        preds += 1

        batch['probas'] = probas.detach()
        batch['confs'] = confidences.detach()
        batch['preds'] = preds.detach()
        batch['logits'] = logits.detach()

        stat_scores = torchmetrics.stat_scores(
            preds,
            labels,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else None,
            mdmc_reduce='global',
            reduce='macro',
            num_classes=self.num_classes
        )

        onehot_labels = self.onehot(labels).float() # B,C,H,W
        final_labels = onehot_labels[:, 1:, ...]
        mask = torch.ones_like(
            final_labels,
            dtype=onehot_labels.dtype,
            device=onehot_labels.device
        )
               
        bce = self.bce(logits, final_labels)
        bce = torch.sum(mask * bce) / torch.sum(mask)
        dice = self.dice(logits * mask, final_labels * mask)
        loss = bce + dice
        
        self.log('Val_BCE', bce)
        self.log('Val_Dice', dice)
        self.log('Val_loss', loss)

        return {'batch': batch,
                'stat_scores': stat_scores.detach(),
                }    
