import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

import dl_toolbox.augmentations as augmentations
from dl_toolbox.utils import TorchOneHot
from dl_toolbox.networks import NetworkFactory


class Supervised(pl.LightningModule):

    def __init__(
        self,
        initial_lr,
        ttas,
        network,
        num_classes,
        class_weights,
        mixup,
        *args,
        **kwargs
    ):

        super().__init__()
        
        self.network = network
        self.num_classes = num_classes
        
        self.loss = nn.CrossEntropyLoss(
            weight=torch.Tensor(class_weights)
        )
        
        self.onehot = TorchOneHot(range(self.num_classes))
        self.mixup = augmentations.Mixup(alpha=mixup) if mixup > 0. else None
        self.ttas = [(augmentations.aug_dict[t](p=1), augmentations.anti_aug_dict[t](p=1)) for t in ttas]
        self.save_hyperparameters(ignore=['network'])
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

    def training_step(self, batch, batch_idx):

        batch = batch['sup']
        inputs = batch['image']
        labels = batch['label']
        if self.mixup:
            labels = self.onehot(labels).float()
            inputs, labels = self.mixup(inputs, labels)
            #batch['image'] = inputs
        logits = self.network(inputs)
        loss = self.loss(logits, labels)
        self.log('Train_sup_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['label']
        logits = self.forward(inputs)
        loss = self.loss(logits, labels)
        self.log('Val_loss', loss)
        
        return logits
    
    def test_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['label']
        logits = self.forward(inputs)
        
        if self.ttas:
            for tta, reverse in self.ttas:
                aux, _ = tta(img=inputs)
                aux_logits = self.forward(aux)
                tta_logits, _ = reverse(img=aux_logits)
                logits = torch.stack([logits, tta_logits])
            logits = logits.mean(dim=0)
        
        loss = self.loss(logits, labels)
        self.log('Test_loss', loss)
        
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