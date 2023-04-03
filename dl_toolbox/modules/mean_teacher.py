import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from copy import deepcopy

import dl_toolbox.augmentations as augmentations
from dl_toolbox.networks import NetworkFactory

class MeanTeacher(pl.LightningModule):

    def __init__(
        self,
        initial_lr,
        ttas,
        network,
        num_classes,
        class_weights,
        alpha_ramp,
        pseudo_threshold,
        consist_aug,
        ema_ramp,
        *args,
        **kwargs
    ):

        super().__init__()
        
        self.network = network
        self.num_classes = num_classes
        self.teacher_network = deepcopy(self.network)
        
        self.loss = nn.CrossEntropyLoss(
            weight=torch.Tensor(class_weights)
        )
        
        self.unsup_loss = nn.CrossEntropyLoss(
            reduction='none'
        )
        self.alpha_ramp = alpha_ramp
        self.pseudo_threshold = pseudo_threshold
        self.consist_aug = augmentations.get_transforms(consist_aug)
        self.ema_ramp = ema_ramp
        
        self.ttas = [(augmentations.aug_dict[t](p=1), augmentations.anti_aug_dict[t](p=1)) for t in ttas]
        self.save_hyperparameters()
        self.initial_lr = initial_lr
        
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)

        return self.optimizer

    def forward(self, x):
        
        return self.network(x)

    def logits2probas(cls, logits):

        return logits.softmax(dim=1)
    
    def probas2confpreds(cls, probas):
        
        return torch.max(probas, dim=1)

    def on_train_epoch_start(self):
        
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log('Prop unsup train', self.alpha)
        self.ema = self.ema_ramp(self.trainer.current_epoch)
        self.log('Ema', self.ema)

    def training_step(self, batch, batch_idx):
        
        batch, unsup_batch = batch["sup"], batch["unsup"]

        inputs = batch['image']
        labels = batch['label']
        logits = self.network(inputs)
        loss = self.loss(logits, labels)
        self.log('Train_sup_loss', loss)
        
        mt_loss = 0.
        if self.alpha > 0.:
            
            unsup_inputs = unsup_batch['image']
            with torch.no_grad():
                teacher_logits = self.teacher_network(unsup_inputs)
            teacher_probas = self.logits2probas(teacher_logits)
            mt_inputs, mt_probas = self.consist_aug(
                img=unsup_inputs,
                label=teacher_probas
            )
            mt_confs, mt_preds = self.probas2confpreds(mt_probas)
            mt_certain = (mt_confs > self.pseudo_threshold).float()
            mt_certain_sum = torch.sum(mt_certain) + 1e-5
            self.log('MT certainty prop', torch.mean(mt_certain))
            mt_logits = self.network(mt_inputs)
            mt_loss = self.unsup_loss(mt_logits, mt_preds)
            mt_loss = torch.sum(mt_certain * mt_loss) / mt_certain_sum
            
        self.log('MT loss', mt_loss)
        loss += self.alpha * mt_loss
        self.log("Train_loss", loss)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

    def validation_step(self, batch, batch_idx):
        
        inputs = batch['image']
        labels = batch['label']
        logits = self.forward(inputs)
        loss = self.loss(logits, labels)
        self.log('Val_loss', loss)

        teacher_logits = self.teacher_network(inputs)
        teacher_probas = self.logits2probas(teacher_logits)
    
        # Validation unsup loss
        mt_inputs, mt_probas = self.consist_aug(
            img=inputs,
            label=teacher_probas
        )
        mt_confs, mt_preds = self.probas2confpreds(mt_probas)
        mt_certain = (mt_confs > self.pseudo_threshold).float()
        mt_certain_sum = torch.sum(mt_certain) + 1e-5
        self.log('Val prop of confident teacher labels', torch.mean(mt_certain))
        mt_logits = self.network(mt_inputs)
        mt_loss = self.unsup_loss(mt_logits, mt_preds)
        mt_loss = torch.sum(mt_certain * mt_loss) / mt_certain_sum
        self.log('Val_mt_loss', mt_loss)
        
        # Checking pseudo_labels quality
        teacher_confs, teacher_preds = self.probas2confpreds(teacher_probas)
        teacher_certain = (teacher_confs > self.pseudo_threshold).float()
        teacher_certain_sum = torch.sum(teacher_certain) + 1e-5
        accus = teacher_preds.eq(labels).float()
        teacher_accus = torch.sum(teacher_certain * accus) / teacher_certain_sum
        self.log('Val acc of teacher labels', teacher_accus)

        return logits

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
