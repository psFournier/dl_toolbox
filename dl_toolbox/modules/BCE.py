import torch
import torch.nn as nn
import pytorch_lightning as pl

from dl_toolbox.utils import TorchOneHot
from dl_toolbox.losses import DiceLoss

class Multilabel(pl.LightningModule):

    def __init__(
        self,
        network,
        optimizer,
        scheduler,
        class_weights,
        in_channels,
        num_classes,
        #mixup,
        *args,
        **kwargs
    ):
        
        super().__init__()
        #self.save_hyperparameters()
        
        self.network = network(
            in_channels=in_channels,
            classes=num_classes-1
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.num_classes = num_classes
        pos_weight = torch.Tensor(class_weights[1:]).reshape(1, num_classes-1, 1, 1)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(
            mode='multilabel',
            log_loss=False,
            from_logits=True,
            smooth=0.01,
            ignore_index=None,
            eps=1e-7
        )
        self.onehot = TorchOneHot(range(1,num_classes))

    def configure_optimizers(self):
        
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)

        return [optimizer], [scheduler]
    
    def forward(self, x):

        return self.network(x)

    def logits2probas(self, logits):
        
        probas = torch.sigmoid(logits)
        confs, preds = torch.max(probas, axis=1)
        nodata_proba = torch.unsqueeze(1 - confs, 1)
        all_probas = torch.cat([nodata_proba, probas], axis=1)
        return all_probas

    def probas2confpreds(self, probas):

        aux_confs, aux_preds = torch.max(probas[:,1:,...], axis=1)
        cond = aux_confs > 0.9
        preds = torch.where(cond, aux_preds+1, 0)
        confs = torch.where(cond, aux_confs, 1-aux_confs)
        
        return confs, preds
    
    def training_step(self, batch, batch_idx):
        
        batch = batch["sup"]
        inputs = batch['image']
        labels = batch['label']
        onehot_labels = self.onehot(labels).float() # B,C or C-1,H,W
        logits = self.network(inputs) # B,C or C-1,H,W
        bce = self.bce(logits, onehot_labels)
        dice = self.dice(logits, onehot_labels)
        loss = bce + dice
        self.log('loss/train', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        
        inputs = batch['image']
        labels = batch['label']
        onehot_labels = self.onehot(labels).float() # B,C or C-1,H,W
        logits = self.forward(inputs)        
        bce = self.bce(logits, onehot_labels)
        dice = self.dice(logits, onehot_labels)
        loss = bce + dice
        self.log('loss/val', loss)
        
        return logits
    
    def predict_step(self, batch, batch_idx):
        
        inputs = batch['image']
        logits = self.forward(inputs)
        
        return logits

    
#class BCE(pl.LightningModule):
#
#    def __init__(
#        self,
#        initial_lr,
#        ttas,
#        network,
#        weights,
#        no_pred_zero,
#        mixup=0.,
#        *args,
#        **kwargs
#    ):
#
#        super().__init__()
#        
#        self.net_factory = NetworkFactory()
#        net_cls = self.net_factory.create(network)
#        self.network = net_cls(*args, **kwargs)
#        
#        num_classes = self.network.out_channels
#        self.no_pred_zero = no_pred_zero
#        weights = torch.Tensor(weights).reshape(1, -1, *self.network.out_dim) if len(weights)>0 else None
#        self.loss = nn.BCEWithLogitsLoss(pos_weight=weights)
#        
#        self.onehot = TorchOneHot(range(num_classes)) # To change for no_pred_zero ?
#        self.mixup = augmentations.Mixup(alpha=mixup) if mixup > 0. else None
#        self.ttas = [(aug_dict[t](p=1), anti_aug_dict[t](p=1)) for t in ttas]
#        self.save_hyperparameters()
#        self.initial_lr = initial_lr
#
#    def forward(self, x):
#
#        return self.network(x)
#
#    def logits2probas(self, logits):
#
#        return torch.sigmoid(logits)
#
#    def probas2confpreds(self, probas):
#
#        aux_confs, aux_preds = torch.max(probas, axis=1)
#        if self.no_pred_zero:
#            cond = aux_confs > 0.5
#            preds = torch.where(cond, aux_preds + 1, 0)
#            confs = torch.where(cond, aux_confs, 1-aux_confs)
#            return confs, preds
#        return aux_confs, aux_preds
#
#    def training_step(self, batch, batch_idx):
#        
#        batch = batch["sup"]
#        inputs = batch['image']
#        labels = batch['mask']
#        labels = self.onehot(labels).float() # B,C or C-1,H,W
#        if self.mixup:
#            inputs, labels = self.mixup(inputs, onehot_labels)
#        logits = self.network(inputs) # B,C or C-1,H,W
#        loss = self.loss(logits, labels)
#        self.log('Train_sup_BCE', loss)
#        batch['logits'] = logits.detach()
#
#        return {'batch': batch, "loss": loss}
#
#    def validation_step(self, batch, batch_idx):
#        
#        inputs = batch['image']
#        labels = batch['label']
#        onehot_labels = self.onehot(labels).float() # B,C or C-1,H,W
#        logits = self.forward(inputs)
#        loss = self.loss(logits, onehot_labels)
#        self.log('Val_BCE', loss)
#        
#        return logits
