import pytorch_lightning as pl
import torch
import torch.nn as nn


class CrossPseudoSupervision(pl.LightningModule):
    def __init__(
        self,
        network1,
        network2,
        optimizer,
        scheduler,
        alpha_ramp,
        class_weights,
        in_channels,
        num_classes,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network1 = network1(in_channels=in_channels, classes=num_classes)
        self.network2 = network2(in_channels=in_channels, classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        self.alpha_ramp = alpha_ramp

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.network1(x)

    def logits2probas(cls, logits):
        return logits.softmax(dim=1)

    def probas2confpreds(cls, probas):
        return torch.max(probas, dim=1)

    def on_train_epoch_start(self):
        self.alpha = self.alpha_ramp(self.trainer.current_epoch)
        self.log("Prop unsup train", self.alpha)

    def training_step(self, batch, batch_idx):
        batch, unsup_batch = batch["sup"], batch["unsup"]
        inputs = batch["image"]
        labels = batch["label"]
        logits1 = self.network1(inputs)
        logits2 = self.network2(inputs)
        loss = (self.ce(logits1, labels) + self.ce(logits2, labels)) / 2
        self.log("ce/train", loss)

        with torch.no_grad():
            _, sup_pl_2 = torch.max(logits2, dim=1)
        with torch.no_grad():
            _, sup_pl_1 = torch.max(logits1, dim=1)
        pl_acc = sup_pl_1.eq(labels).float().mean()
        sup_cps_loss = self.ce(logits1, sup_pl_2) + self.ce(logits2, sup_pl_1)
        self.log("pseudolabel accuracy/train", pl_acc)
        self.log("cps_sup/train", sup_cps_loss)

        unsup_cps_loss = 0.0
        if self.alpha > 0.0:
            unsup_inputs = unsup_batch["image"]
            unsup_logits_1 = self.network1(unsup_inputs)
            unsup_logits_2 = self.network2(unsup_inputs)
            with torch.no_grad():
                _, unsup_pl_2 = torch.max(unsup_logits_2, dim=1)
            unsup_pl_2_loss = self.ce(unsup_logits_1, unsup_pl_2)
            with torch.no_grad():
                _, unsup_pl_1 = torch.max(unsup_logits_1, dim=1)
            unsup_pl_1_loss = self.ce(unsup_logits_2, unsup_pl_1)
            unsup_cps_loss = unsup_pl_1_loss + unsup_pl_2_loss
        self.log("cps_unsup/train", unsup_cps_loss)
        
        loss += self.alpha * sup_cps_loss
        loss += self.alpha * unsup_cps_loss
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        logits1 = self.network1(inputs)
        ce = self.ce(logits1, labels)
        self.log("ce/val", ce)
        
        _, pl_1 = torch.max(logits1, dim=1)
        pl_acc = pl_1.eq(labels).float().mean()
        self.log("pseudolabel accuracy/val", pl_acc)
        return logits1

    def predict_step(self, batch, batch_idx):
        inputs = batch["image"]
        logits = self.forward(inputs)
        return logits
    
    
            ## Supervising network 2 with pseudolabels from network 1
            # with torch.no_grad():
            #    pseudo_probs_1 = self.logits2probas(unsup_logits_1)
            # pseudo_confs_1, pseudo_preds_1 = self.probas2confpreds(pseudo_probs_1)
            # loss_no_reduce_2 = self.unsup_loss(
            #    unsup_logits_2,
            #    pseudo_preds_1
            # ) # B,H,W
            # pseudo_certain_1 = (pseudo_confs_1 > self.pseudo_threshold).float() # B,H,W
            # pseudo_certain_1_sum = torch.sum(pseudo_certain_1) + 1e-5
            # self.log('Pseudo_certain_1', torch.mean(pseudo_certain_1))
            # pseudo_loss_2 = torch.sum(pseudo_certain_1 * loss_no_reduce_2) / pseudo_certain_1_sum