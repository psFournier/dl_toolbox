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



class BCE(BaseModule):

    # BCE_multilabel = Binary Cross Entropy for multilabel prediction

    def __init__(self,
                 network,
                 weights,
                 pred_zero,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        net_cls = self.net_factory.create(network)
        self.network = net_cls(*args, **kwargs)
        self.num_classes = self.network.out_channels  
        self.weights = list(weights) if len(weights)>0 else [1]*self.num_classes
        self.ignore_index = -1
        self.loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(self.weights).reshape(1, -1, 1, 1)
        )
        self.pred_zero = pred_zero
        self.onehot = TorchOneHot(
            range(~self.pred_zero, self.num_classes+~self.pred_zero)
        )
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--network", type=str)
        parser.add_argument("--weights", type=float, nargs="+", default=())
        parser.add_argument("--pred_zero", type=bool)
        
        return parser

    def forward(self, x):
        
        return self.network(x)
    
    def _compute_probas(self, logits):

        return torch.sigmoid(logits)
    
    def _compute_conf_preds(self, probas):
        
        aux_confs, aux_preds = torch.max(probas, axis=1)
        if self.pred_zero:
            return aux_confs, aux_preds
        else:
            cond = aux_confs > 0.5
            preds = torch.where(cond, aux_preds + 1, 0)
            confs = torch.where(cond, aux_confs, 1-aux_confs)
            return confs, preds
        
    def training_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        onehot_labels = self.onehot(labels).float() # B,C or C-1,H,W
        logits = self.network(inputs) # B,C or C-1,H,W
        loss = self.loss(logits, onehot_labels)
        self.log('Train_sup_BCE', loss)
        batch['logits'] = logits.detach()

        return {'batch': batch, "loss": loss}


    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        labels = batch['mask']
        logits = outs['logits']
        onehot_labels = self.onehot(labels).float() # B,C or C-1,H,W
        loss = self.loss(logits, onehot_labels)
        self.log('Val_BCE', loss)

        return outs    
