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
                 ignore_index,
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
        self.out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.ignore_index = list(ignore_index) if len(ignore_index)>0 else [-1]*self.num_classes
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
        parser.add_argument("--ignore_index", type=int, nargs="+", default=())
        
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
        for i, idx in enumerate(self.ignore_index[1:]):
            if idx >= 0:
                mask[:, i, ...] -= onehot_labels[:, idx, ...]
            
        logits = self.network(inputs) # B,C-1,H,W
        bce = self.bce(logits, final_labels)
        bce = torch.sum(mask * bce) / torch.sum(mask)
        #dice = self.dice(logits * mask, final_labels * mask)
        dice = 0
        loss = bce + dice
        
        self.log('Train_sup_BCE', bce)
        self.log('Train_sup_Dice', dice)
        self.log('Train_sup_loss', loss)

        batch['logits'] = logits.detach()

        return {'batch': batch, "loss": loss}
    
    def _compute_probas(self, logits):

        return torch.sigmoid(logits)
    
    def _compute_conf_preds(self, probas):
        
        aux_confs, aux_preds = torch.max(probas, axis=1)
        cond = aux_confs > 0.5
        preds = torch.where(cond, aux_preds + 1, 0)
        confs = torch.where(cond, aux_confs, 1-aux_confs)
        
        return confs, preds

    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        
        labels = batch['mask']
        logits = outs['logits']
        onehot_labels = self.onehot(labels).float() # B,C,H,W
        final_labels = onehot_labels[:, 1:, ...]
        mask = torch.ones_like(
            final_labels,
            dtype=onehot_labels.dtype,
            device=onehot_labels.device
        )
        for i, idx in enumerate(self.ignore_index[1:]):
            if idx >= 0:
                mask[:, i, ...] -= onehot_labels[:, idx, ...]
               
        bce = self.bce(logits, final_labels)
        bce = torch.sum(mask * bce) / torch.sum(mask)
        #dice = self.dice(logits * mask, final_labels * mask)
        dice = 0
        loss = bce + dice
        
        self.log('Val_BCE', bce)
        self.log('Val_Dice', dice)
        self.log('Val_loss', loss)

        return outs    
