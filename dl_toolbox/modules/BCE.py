import pytorch_lightning as pl
import torch
import torch.nn as nn

from dl_toolbox.losses import DiceLoss
#from dl_toolbox.utils import TorchOneHot


class Multilabel(pl.LightningModule):
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
        self.network = network(in_channels=in_channels, classes=num_classes - 1)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        pos_weight = torch.Tensor(class_weights[1:]).view(1, num_classes - 1, 1, 1)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(
            mode="multilabel",
            log_loss=False,
            from_logits=True,
            smooth=0.01,
            ignore_index=None,
            eps=1e-7,
        )

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
        aux_confs, aux_preds = torch.max(probas[:, 1:, ...], axis=1)
        cond = aux_confs > 0.6
        preds = torch.where(cond, aux_preds + 1, 0)
        confs = torch.where(cond, aux_confs, 1 - aux_confs)
        return confs, preds

    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        inputs = batch["image"]
        labels = batch["label"]
        onehot_labels = nn.functional.one_hot(labels).float()[:,1:,...]
        #onehot_labels = self.onehot(labels).float()  # B,C or C-1,H,W
        logits = self.network(inputs)  # B,C or C-1,H,W
        bce = self.bce(logits, onehot_labels)
        dice = self.dice(logits, onehot_labels)
        loss = bce + dice
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        #onehot_labels = self.onehot(labels).float()  # B,C or C-1,H,W
        onehot_labels = nn.functional.one_hot(labels).float()[:,1:,...]
        logits = self.forward(inputs)
        bce = self.bce(logits, onehot_labels)
        dice = self.dice(logits, onehot_labels)
        loss = bce + dice
        self.log("loss/val", loss)
        return logits

    def predict_step(self, batch, batch_idx):
        inputs = batch["image"]
        logits = self.forward(inputs)
        return logits