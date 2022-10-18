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

class CPS(BaseModule):

    # CPS = Cross Pseudo Supervision

    def __init__(self,
                 encoder,
                 in_channels,
                 final_alpha,
                 alpha_milestones,
                 pseudo_threshold,
                 ignore_index,
                 pretrained=True,
                 initial_lr=0.05,
                 final_lr=0.001,
                 lr_milestones=(0.5,0.9),
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.network1 = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=self.num_classes,
            decoder_use_batchnorm=True
        )

        self.network2 = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=self.num_classes,
            decoder_use_batchnorm=True
        )

        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)

        self.loss1 = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            weight=torch.Tensor(self.weights)
        )
        self.loss2 = DiceLoss(
            mode="multiclass",
            log_loss=False,
            from_logits=True,
            ignore_index=self.ignore_index
        )

        self.unsup_loss = nn.CrossEntropyLoss(
            reduction='none'
        )

        self.final_alpha = final_alpha
        self.alpha_milestones = alpha_milestones
        self.alpha = 0.
        self.pseudo_threshold = pseudo_threshold
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--final_alpha", type=float)
        parser.add_argument("--alpha_milestones", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)
        parser.add_argument("--ignore_index", type=int)

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
            
    def forward(self, x):

        logits1 = self.network1(x)
        logits2 = self.network2(x)
        
        return (logits1 + logits2) / 2

    def training_step(self, batch, batch_idx):

        batch, unsup_batch = batch["sup"], batch["unsup"]

        inputs = batch['image']
        labels = batch['mask']

        logits1 = self.network1(inputs)
        logits2 = self.network2(inputs)
        logits = (logits1 + logits2) / 2
        loss1 = self.loss1(logits1, labels) 
        loss1 += self.loss1(logits2, labels)
        loss1 /= 2
        #loss2 = self.loss2(logits1, labels)
        #loss2 += self.loss2(logits2, labels)
        loss2=0
        loss2 /= 2
        loss = loss1 + loss2

        batch['logits'] = logits.detach()
        outputs = {'batch': batch}
 
        # Supervising network 1 with pseudolabels from network 2
            
        #pseudo_probs_2 = logits2.detach().softmax(dim=1)
        #top_probs_2, pseudo_preds_2 = torch.max(pseudo_probs_2, dim=1) # B,H,W
        #loss_no_reduce_1 = self.unsup_loss(
        #    logits1,
        #    pseudo_preds_2
        #) # B,H,W
        #pseudo_certain_2 = (top_probs_2 > self.pseudo_threshold).float()
        #certain_2 = torch.sum(pseudo_certain_2)
        #self.log('Pseudo_certain_2_sup', torch.mean(pseudo_certain_2))
        #pseudo_loss_1 = torch.sum(pseudo_certain_2 * loss_no_reduce_1) / certain_2

        # Supervising network 2 with pseudolabels from network 1

        #pseudo_probs_1 = logits1.detach().softmax(dim=1)
        #top_probs_1, pseudo_preds_1 = torch.max(pseudo_probs_1, dim=1)
        #loss_no_reduce_2 = self.unsup_loss(
        #    logits2,
        #    pseudo_preds_1
        #)
        #pseudo_certain_1 = (top_probs_1 > self.pseudo_threshold).float()
        #certain_1 = torch.sum(pseudo_certain_1)
        #pseudo_loss_2 = torch.sum(pseudo_certain_1 * loss_no_reduce_2) / certain_1

        #pseudo_loss_sup = (pseudo_loss_1 + pseudo_loss_2) / 2
        #pseudo_loss = pseudo_loss_sup
        pseudo_loss = 0

        self.log('Train_sup_CE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)
        #self.log('Pseudo_loss_sup', pseudo_loss_sup)

        if self.trainer.current_epoch > self.alpha_milestones[0]:

            unsup_inputs = unsup_batch['image']
            unsup_outputs_1 = self.network1(unsup_inputs)
            unsup_outputs_2 = self.network2(unsup_inputs)
            unsup_logits = (unsup_outputs_1 + unsup_outputs_2) / 2
            unsup_batch['logits'] = unsup_logits.detach()
            outputs['unsup_batch'] = unsup_batch

            # Supervising network 1 with pseudolabels from network 2
            
            pseudo_probs_2 = unsup_outputs_2.detach().softmax(dim=1)
            top_probs_2, pseudo_preds_2 = torch.max(pseudo_probs_2, dim=1)
            loss_no_reduce_1 = self.unsup_loss(
                unsup_outputs_1,
                pseudo_preds_2
            ) # B,H,W
            pseudo_certain_2 = (top_probs_2 > self.pseudo_threshold).float() # B,H,W
            certain_2 = torch.sum(pseudo_certain_2)
            self.log('Pseudo_certain_2_unsup', torch.mean(pseudo_certain_2))
            pseudo_loss_1 = torch.sum(pseudo_certain_2 * loss_no_reduce_1) / certain_2

            # Supervising network 2 with pseudolabels from network 1

            pseudo_probs_1 = unsup_outputs_1.detach().softmax(dim=1)
            top_probs_1, pseudo_preds_1 = torch.max(pseudo_probs_1, dim=1)
            loss_no_reduce_2 = self.unsup_loss(
                unsup_outputs_2,
                pseudo_preds_1
            )
            pseudo_certain_1 = (top_probs_1 > self.pseudo_threshold).float()
            certain_1 = torch.sum(pseudo_certain_1)
            pseudo_loss_2 = torch.sum(pseudo_certain_1 * loss_no_reduce_2) / certain_1

            pseudo_loss_unsup = (pseudo_loss_1 + pseudo_loss_2) / 2
            pseudo_loss += pseudo_loss_unsup
            self.log('Pseudo_loss_unsup', pseudo_loss_unsup)

        self.log('Pseudo label loss', pseudo_loss)
        loss += self.alpha * pseudo_loss
        outputs['loss'] = loss

        self.log('Prop unsup train', self.alpha)
        self.log("Train_loss", loss)

        return outputs
    
    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)

        loss1 = self.loss1(outs['logits'], batch['mask'])
        #loss2 = self.loss2(logits, labels)
        loss2=0
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)
        
        logits1 = self.network1(batch['image'])
        logits2 = self.network2(batch['image'])
        
        pseudo_probs_2 = logits2.detach().softmax(dim=1)
        top_probs_2, pseudo_preds_2 = torch.max(pseudo_probs_2, dim=1)
        pseudo_certain_2 = (top_probs_2 > self.pseudo_threshold).float() # B,H,W
        certain_2 = torch.sum(pseudo_certain_2)
        self.log('Val prop of confident pseudo labels', torch.mean(pseudo_certain_2))
        
        loss_no_reduce_1 = self.unsup_loss(
            logits1,
            pseudo_preds_2
        ) # B,H,W
        pseudo_loss_1 = torch.sum(pseudo_certain_2 * loss_no_reduce_1) / certain_2
        self.log('Val_pseudo_loss_1', pseudo_loss_1)
        
        accus = pseudo_preds_2.eq(batch['mask']).float()
        pseudo_accus = torch.sum(pseudo_certain_2 * accus) / certain_2
        self.log('Val acc of pseudo labels', pseudo_accus)

        return outs