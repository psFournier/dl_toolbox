import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

import dl_toolbox.augmentations as augmentations
from dl_toolbox.networks import NetworkFactory

class CrossPseudoSupervision(pl.LightningModule):

    def __init__(
        self,
        initial_lr,
        ttas,
        network,
        weights,
        alpha_ramp,
        pseudo_threshold,
        *args,
        **kwargs
    ):

        super().__init__()
        
        self.net_factory = NetworkFactory()
        net_cls = self.net_factory.create(network)
        self.network1 = net_cls(*args, **kwargs)
        self.network2 = net_cls(*args, **kwargs)
        
        num_classes = self.network1.out_channels
        weights = list(weights) if len(weights)>0 else [1]*num_classes
        self.loss = nn.CrossEntropyLoss(
            weight=torch.Tensor(weights)
        )
        
        self.unsup_loss = nn.CrossEntropyLoss(
            reduction='none'
        )
        self.alpha_ramp = alpha_ramp
        self.pseudo_threshold = pseudo_threshold
        
        self.ttas = [(augmentations.aug_dict[t](p=1), augmentations.anti_aug_dict[t](p=1)) for t in ttas]
        self.save_hyperparameters()
        self.initial_lr = initial_lr
        
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)

        return self.optimizer

    def forward(self, x):

        logits1 = self.network1(x)
        logits2 = self.network2(x)
        
        return (logits1 + logits2) / 2

    def logits2probas(cls, logits):

        return logits.softmax(dim=1)
    
    def probas2confpreds(cls, probas):
        
        return torch.max(probas, dim=1)
    
    def on_train_epoch_start(self):
        
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)   
        self.log('Prop unsup train', self.alpha)
    
    def training_step(self, batch, batch_idx):

        batch, unsup_batch = batch["sup"], batch["unsup"]

        inputs = batch['image']
        labels = batch['label']

        logits1 = self.network1(inputs)
        logits2 = self.network2(inputs)
        loss = (self.loss(logits1, labels)+self.loss(logits2, labels))/2
        self.log('Train_sup_loss', loss)
        
        if self.alpha > 0.:

            unsup_inputs = unsup_batch['image']
            unsup_logits_1 = self.network1(unsup_inputs)
            unsup_logits_2 = self.network2(unsup_inputs)
            
            # Supervising network 1 with pseudolabels from network 2
            
            with torch.no_grad():
                pseudo_probs_2 = self.logits2probas(unsup_logits_2)
            pseudo_confs_2, pseudo_preds_2 = self.probas2confpreds(pseudo_probs_2)
            loss_no_reduce_1 = self.unsup_loss(
                unsup_logits_1,
                pseudo_preds_2
            ) # B,H,W
            pseudo_certain_2 = (pseudo_confs_2 > self.pseudo_threshold).float() # B,H,W
            pseudo_certain_2_sum = torch.sum(pseudo_certain_2) + 1e-5
            self.log('Pseudo_certain_2', torch.mean(pseudo_certain_2))
            pseudo_loss_1 = torch.sum(pseudo_certain_2 * loss_no_reduce_1) / pseudo_certain_2_sum

            # Supervising network 2 with pseudolabels from network 1
            
            with torch.no_grad():
                pseudo_probs_1 = self.logits2probas(unsup_logits_1)
            pseudo_confs_1, pseudo_preds_1 = self.probas2confpreds(pseudo_probs_1)
            loss_no_reduce_2 = self.unsup_loss(
                unsup_logits_2,
                pseudo_preds_1
            ) # B,H,W
            pseudo_certain_1 = (pseudo_confs_1 > self.pseudo_threshold).float() # B,H,W
            pseudo_certain_1_sum = torch.sum(pseudo_certain_1) + 1e-5
            self.log('Pseudo_certain_1', torch.mean(pseudo_certain_1))
            pseudo_loss_2 = torch.sum(pseudo_certain_1 * loss_no_reduce_2) / pseudo_certain_1_sum
            
            cps_loss = (pseudo_loss_1 + pseudo_loss_2) / 2

        self.log('cps loss', cps_loss)
        loss += self.alpha * cps_loss
        self.log("Train_loss", loss)
        
        return loss

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


    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['label']
        logits1 = self.network1(inputs)
        loss = self.loss(logits1, labels)
        self.log('Val_sup_loss', loss)
        
        # Validation unsup loss
        logits2 = self.network2(inputs)
        pseudo_probs_2 = self.logits2probas(logits2)
        pseudo_confs_2, pseudo_preds_2 = self.probas2confpreds(pseudo_probs_2)
        pseudo_certain_2 = (pseudo_confs_2 > self.pseudo_threshold).float() # B,H,W
        pseudo_certain_2_sum = torch.sum(pseudo_certain_2)
        self.log('Val prop of confident pseudo labels', torch.mean(pseudo_certain_2))
        loss_no_reduce_1 = self.unsup_loss(
            logits1,
            pseudo_preds_2
        ) # B,H,W
        pseudo_loss_1 = torch.sum(pseudo_certain_2 * loss_no_reduce_1) / pseudo_certain_2_sum
        self.log('Val_pseudo_loss_1', pseudo_loss_1)
        
        # Validation quality of pseudo labels
        accus = pseudo_preds_2.eq(labels).float()
        pseudo_accus = torch.sum(pseudo_certain_2 * accus) / pseudo_certain_2_sum
        self.log('Val acc of pseudo labels', pseudo_accus)
                
        return logits1

    def predict_step(self, batch, batch_idx):
        
        inputs = batch['image']
        logits = self.forward(inputs)
        
        if self.ttas:
            for tta, reverse in self.ttas:
                aux, _ = tta(img=inputs)
                aux_logits = self.forward(aux)
                tta_logits, _ = reverse(img=aux_logits)
                logits = torch.stack([logits, tta_logits])
            logits = logits.mean(dim=0)
        
        return logits