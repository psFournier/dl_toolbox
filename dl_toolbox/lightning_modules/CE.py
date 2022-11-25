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


class CE(BaseModule):

    def __init__(self,
                 network,
                 weights,
                 ignore_index,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        net_cls = self.net_factory.create(network)
        self.network = net_cls(*args, **kwargs)
        self.num_classes = self.network.out_channels
        out_dim = self.network.out_dim
        self.weights = list(weights) if len(weights)>0 else [1]*self.num_classes
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            weight=torch.Tensor(self.weights)
            #weight=torch.Tensor(self.weights).reshape(1,-1,*out_dim)
        )
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ignore_index", type=int)
        parser.add_argument("--network", type=str)
        parser.add_argument("--weights", type=float, nargs="+", default=())

        return parser

    def forward(self, x):
        
        return self.network(x)

    @classmethod
    def _compute_probas(cls, logits):

        return logits.softmax(dim=1)
    
    def _compute_conf_preds(self, probas):
        
        return torch.max(probas, dim=1)

    def training_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.network(inputs)
        loss = self.loss(logits, labels)
        self.log('Train_sup_CE', loss)
        batch['logits'] = logits.detach()

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        labels = batch['mask']
        logits = outs['logits']
        loss = self.loss(logits, labels)
        self.log('Val_CE', loss)

        return outs    
