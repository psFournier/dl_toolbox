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
        network2,
        num_classes,
        class_weights,
        alpha_ramp,
        pseudo_threshold,
        *args,
        **kwargs
    ):

        super().__init__()
        
        self.num_classes = num_classes
        self.network1 = network
        self.network2 = network2
        
        self.loss = nn.CrossEntropyLoss(
            weight=torch.Tensor(class_weights)
        )
        
        self.unsup_loss = nn.CrossEntropyLoss(
            weight=torch.Tensor(class_weights),
            #reduction='none'
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
        
        return self.network1(x)

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
    
        with torch.no_grad():
            _, sup_pl_2 = torch.max(logits2, dim=1)
        sup_pl_2_loss = self.unsup_loss(logits1, sup_pl_2)
        with torch.no_grad():
            _, sup_pl_1 = torch.max(logits1, dim=1)
        sup_pl_1_loss = self.unsup_loss(logits2, sup_pl_1)
        sup_cps_loss = sup_pl_1_loss + sup_pl_2_loss
        loss += self.alpha * sup_cps_loss
        self.log('Train_sup_cps_loss', sup_cps_loss)
           
        unsup_cps_loss = 0.
        if self.alpha > 0.:

            unsup_inputs = unsup_batch['image']
            unsup_logits_1 = self.network1(unsup_inputs)
            unsup_logits_2 = self.network2(unsup_inputs)
                        
            with torch.no_grad():
                _, unsup_pl_2 = torch.max(unsup_logits_2, dim=1)
            unsup_pl_2_loss = self.unsup_loss(unsup_logits_1, unsup_pl_2)
            with torch.no_grad():
                _, unsup_pl_1 = torch.max(unsup_logits_1, dim=1)
            unsup_pl_1_loss = self.unsup_loss(unsup_logits_2, unsup_pl_1)
            unsup_cps_loss = unsup_pl_1_loss + unsup_pl_2_loss
            
            #pseudo_confs_2, pseudo_preds_2 = self.probas2confpreds(pseudo_probs_2)
            #loss_no_reduce_1 = self.unsup_loss(
            #    unsup_logits_1,
            #    pseudo_preds_2
            #) # B,H,W
            #pseudo_certain_2 = (pseudo_confs_2 > self.pseudo_threshold).float() # B,H,W
            #pseudo_certain_2_sum = torch.sum(pseudo_certain_2) + 1e-5
            #self.log('Pseudo_certain_2', torch.mean(pseudo_certain_2))
            #pseudo_loss_1 = torch.sum(pseudo_certain_2 * loss_no_reduce_1) / pseudo_certain_2_sum

            # Supervising network 2 with pseudolabels from network 1
            
            #with torch.no_grad():
            #    pseudo_probs_1 = self.logits2probas(unsup_logits_1)
            #pseudo_confs_1, pseudo_preds_1 = self.probas2confpreds(pseudo_probs_1)
            #loss_no_reduce_2 = self.unsup_loss(
            #    unsup_logits_2,
            #    pseudo_preds_1
            #) # B,H,W
            #pseudo_certain_1 = (pseudo_confs_1 > self.pseudo_threshold).float() # B,H,W
            #pseudo_certain_1_sum = torch.sum(pseudo_certain_1) + 1e-5
            #self.log('Pseudo_certain_1', torch.mean(pseudo_certain_1))
            #pseudo_loss_2 = torch.sum(pseudo_certain_1 * loss_no_reduce_2) / pseudo_certain_1_sum
            
            #cps_loss = (pseudo_loss_1 + pseudo_loss_2) / 2

        self.log('Train_unsup_cps_loss', unsup_cps_loss)
        loss += self.alpha * unsup_cps_loss
        self.log("Train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['label']
        logits1 = self.network1(inputs)
        loss = self.loss(logits1, labels)
        self.log('Val_loss', loss)
        
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