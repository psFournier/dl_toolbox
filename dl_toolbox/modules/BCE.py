import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

import dl_toolbox.augmentations as augmentations
from dl_toolbox.utils import TorchOneHot
from dl_toolbox.networks import NetworkFactory
from dl_toolbox.torch_datasets.utils import aug_dict, anti_aug_dict


class BCE(pl.LightningModule):

    def __init__(
        self,
        initial_lr,
        ttas,
        network,
        weights,
        no_pred_zero,
        mixup=0.,
        *args,
        **kwargs
    ):

        super().__init__()
        
        self.net_factory = NetworkFactory()
        net_cls = self.net_factory.create(network)
        self.network = net_cls(*args, **kwargs)
        
        num_classes = self.network.out_channels
        self.no_pred_zero = no_pred_zero
        weights = torch.Tensor(weights).reshape(1, -1, *self.network.out_dim) if len(weights)>0 else None
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weights)
        
        self.onehot = TorchOneHot(range(num_classes)) # To change for no_pred_zero ?
        self.mixup = augmentations.Mixup(alpha=mixup) if mixup > 0. else None
        self.ttas = [(aug_dict[t](p=1), anti_aug_dict[t](p=1)) for t in ttas]
        self.save_hyperparameters()
        self.initial_lr = initial_lr

    def forward(self, x):

        return self.network(x)

    def logits2probas(self, logits):

        return torch.sigmoid(logits)

    def probas2confpreds(self, probas):

        aux_confs, aux_preds = torch.max(probas, axis=1)
        if self.no_pred_zero:
            cond = aux_confs > 0.5
            preds = torch.where(cond, aux_preds + 1, 0)
            confs = torch.where(cond, aux_confs, 1-aux_confs)
            return confs, preds
        return aux_confs, aux_preds

    def training_step(self, batch, batch_idx):
        
        batch = batch["sup"]
        inputs = batch['image']
        labels = batch['mask']
        labels = self.onehot(labels).float() # B,C or C-1,H,W
        if self.mixup:
            inputs, labels = self.mixup(inputs, onehot_labels)
        logits = self.network(inputs) # B,C or C-1,H,W
        loss = self.loss(logits, labels)
        self.log('Train_sup_BCE', loss)
        batch['logits'] = logits.detach()

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):
        
        inputs = batch['image']
        labels = batch['label']
        onehot_labels = self.onehot(labels).float() # B,C or C-1,H,W
        logits = self.forward(inputs)
        loss = self.loss(logits, onehot_labels)
        self.log('Val_BCE', loss)
        
        return logits
