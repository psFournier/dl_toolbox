import torch
import torch.nn as nn
import pytorch_lightning as pl

import dl_toolbox.augmentations as augmentations
from dl_toolbox.utils import TorchOneHot

        
class Supervised(pl.LightningModule):

    def __init__(
        self,
        network,
        optimizer,
        loss,
        class_weights,
        #ttas,
        in_channels,
        num_classes,
        mixup,
        *args,
        **kwargs
    ):

        super().__init__()
        #self.save_hyperparameters()
        
        self.network = network(
            in_channels=in_channels,
            classes=num_classes
        )
        self.optimizer = optimizer
        
        self.num_classes = num_classes
        
        self.loss = loss(
            weight=torch.Tensor(class_weights)
        )
        #self.dice_loss = DiceLoss(mode="multilabel", log_loss=False, from_logits=True)
        
        self.onehot = TorchOneHot(range(self.num_classes))
        #self.mixup = augmentations.Mixup(alpha=mixup) if mixup > 0. else None
        #self.ttas = [(augmentations.aug_dict[t](p=1), augmentations.anti_aug_dict[t](p=1)) for t in ttas]
        
    def configure_optimizers(self):
        
        optimizer = self.optimizer(params=self.parameters())
        
        #optimizer = SGD(
        #    self.parameters(),
        #    lr=self.initial_lr,
        #    momentum=0.9,
        #    weight_decay=1e-4
        #)

        def lr_foo(epoch):
            if epoch <= 5:
                # warm up lr
                lr_scale = epoch / 5
            else:
                lr_scale = (1 - float(epoch) / 100) ** 0.9
            return lr_scale

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )

        return [optimizer], [scheduler]
        

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
        
        #if self.ttas:
        #    for tta, reverse in self.ttas:
        #        aux, _ = tta(img=inputs)
        #        aux_logits = self.forward(aux)
        #        tta_logits, _ = reverse(img=aux_logits)
        #        logits = torch.stack([logits, tta_logits])
        #    logits = logits.mean(dim=0)
        
        loss = self.loss(logits, labels)
        self.log('Test_loss', loss)
        
        return logits

    def predict_step(self, batch, batch_idx):
        
        inputs = batch['image']
        logits = self.forward(inputs)
        
        #if self.ttas:
        #    for tta, reverse in self.ttas:
        #        aux, _ = tta(img=inputs)
        #        aux_logits = self.forward(aux)
        #        tta_logits, _ = reverse(img=aux_logits)
        #        logits = torch.stack([logits, tta_logits])
        #    logits = logits.mean(dim=0)
        
        return logits