import pytorch_lightning as pl
import torch
import torch.nn as nn

from dl_toolbox.losses import DiceLoss


class Supervised(pl.LightningModule):
    def __init__(
        self,
        network,
        optimizer,
        scheduler,
        class_weights,
        in_channels,
        num_classes,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network = network(in_channels, num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.network(x)

    def logits2probas(cls, logits):
        return logits.softmax(dim=1)

    def probas2confpreds(cls, probas):
        return torch.max(probas, dim=1)

    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        inputs = batch["image"]
        labels = batch["label"]
        logits = self.network(inputs)
        ce = self.ce(logits, labels)
        self.log(f"cross_entropy/train", ce)
        return ce

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        logits = self.forward(inputs)
        ce = self.ce(logits, labels)
        self.log(f"cross_entropy/val", ce)
        return logits

    def predict_step(self, batch, batch_idx):
        inputs = batch["image"]
        logits = self.forward(inputs)
        # if self.ttas:
        #    for tta, reverse in self.ttas:
        #        aux, _ = tta(img=inputs)
        #        aux_logits = self.forward(aux)
        #        tta_logits, _ = reverse(img=aux_logits)
        #        logits = torch.stack([logits, tta_logits])
        #    logits = logits.mean(dim=0)
        return logits

class SupervisedDice(Supervised):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dice = DiceLoss(
            mode="multiclass",
            log_loss=False,
            from_logits=True,
            smooth=0.01,
            ignore_index=None,
            eps=1e-7,
        )

    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        inputs = batch["image"]
        labels = batch["label"]
        logits = self.network(inputs)
        ce = self.ce(logits, labels)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits, labels)
        self.log(f"dice/train", dice)
        return ce+dice

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        logits = self.forward(inputs)
        ce = self.ce(logits, labels)
        self.log(f"cross_entropy/val", ce)
        dice = self.dice(logits, labels)
        self.log(f"dice/val", dice)
        return logits