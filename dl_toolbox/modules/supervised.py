import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as M
import matplotlib.pyplot as plt
from dl_toolbox.utils import plot_confusion_matrix


class Supervised(pl.LightningModule):
    def __init__(
        self,
        network,
        in_channels,
        num_classes,
        optimizer,
        scheduler,
        ce_loss,
        dice_loss,
        batch_tf,
        tta,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network = network(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.ce = ce_loss()
        self.dice = dice_loss(mode="multilabel")
        self.batch_tf = batch_tf
        self.tta = tta
        print(num_classes)
        metric_args = {'task':'multiclass', 'num_classes':num_classes, 'num_labels': num_classes, 'ignore_index':self.ce.ignore_index}
        self.train_accuracy = M.Accuracy(**metric_args)
        self.val_accuracy = M.Accuracy(**metric_args)
        self.test_accuracy = M.Accuracy(**metric_args)
        self.val_cm = M.ConfusionMatrix(**metric_args)
        self.test_cm = M.ConfusionMatrix(**metric_args)
        self.val_jaccard = M.JaccardIndex(**metric_args)
        self.test_jaccard = M.JaccardIndex(**metric_args)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler(optimizer=optimizer),
                "interval": "epoch"
            },
        }

    def forward(self, x):
        return self.network(x)

    def logits2probas(cls, logits):
        return logits.softmax(dim=1)

    def probas2confpreds(cls, probas):
        return torch.max(probas, dim=1)   
    
    #def probas2confpreds(self, probas):
    #    aux_confs, aux_preds = torch.max(probas[:, 1:, ...], axis=1)
    #    cond = aux_confs > 0.6
    #    preds = torch.where(cond, aux_preds + 1, 0)
    #    confs = torch.where(cond, aux_confs, 1 - aux_confs)
    #    return confs, preds
        
    def on_train_epoch_end(self):
        self.log("accuracy/train", self.train_accuracy.compute())
        self.train_accuracy.reset()
        
    def one_hot(self, y):
        return F.one_hot(y.unsqueeze(1), self.num_classes).transpose(1,-1).squeeze().float()
    
    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        xs = batch["image"]
        ys = self.one_hot(batch["label"])
        xs, ys = self.batch_tf(xs, ys)
        logits_xs = self.network(xs)
        ce = self.ce(logits_xs, ys)
        self.log(f"cross_entropy/train", ce)
        dice = self.dice(logits_xs, ys)
        self.log(f"dice/train", dice)
        #_, preds = self.probas2confpreds(self.logits2probas(logits_xs))
        #self.train_accuracy.update(preds, batch["label"])
        return ce + dice
        
    def validation_step(self, batch, batch_idx):
        xs = batch["image"]
        ys = self.one_hot(batch["label"])
        logits_xs = self.forward(xs)
        ce = self.ce(logits_xs, ys)
        self.log(f"cross_entropy/val", ce)
        dice = self.dice(logits_xs, ys)
        self.log(f"dice/val", dice)
        _, preds = self.probas2confpreds(self.logits2probas(logits_xs))
        self.val_accuracy.update(preds, batch["label"])
        self.val_cm.update(preds, batch["label"])
        self.val_jaccard.update(preds, batch["label"])
        
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        self.log("iou/val", self.val_jaccard.compute())
        confmat = self.val_cm.compute().detach().cpu()
        class_names = self.trainer.datamodule.class_names
        logger = self.trainer.logger
        if logger:
            fig1 = plot_confusion_matrix(confmat, class_names, "recall", fontsize=8)
            logger.experiment.add_figure("Recall matrix", fig1, global_step=self.trainer.global_step)
            fig2 = plot_confusion_matrix(confmat, class_names, "precision", fontsize=8)
            logger.experiment.add_figure("Precision matrix", fig2, global_step=self.trainer.global_step)
        self.val_accuracy.reset()
        self.val_jaccard.reset()
        self.val_cm.reset()

    def test_step(self, batch, batch_idx):
        xs = batch["image"]
        ys = self.one_hot(batch["label"])
        logits_xs = self.forward(xs)
        if self.tta is not None:
            auxs = [self.forward(x) for x in self.tta(xs)]
            logits_xs = torch.stack([logits_xs] + self.tta.revert(auxs)).sum(dim=0)
        probas = self.logits2probas(logits_xs)
        _, preds = self.probas2confpreds(probas)
        self.test_accuracy.update(preds, batch["label"])
        self.test_jaccard.update(preds, batch["label"])
        self.test_cm.update(preds, batch["label"])
        
    def on_test_epoch_end(self):
        self.log("accuracy/test", self.test_accuracy.compute())
        self.log("iou/test", self.test_jaccard.compute())
        confmat = self.test_cm.compute().detach().cpu()
        self.test_accuracy.reset()
        self.test_jaccard.reset()
        self.test_cm.reset()
        class_names = self.trainer.datamodule.class_names
        fig = plot_confusion_matrix(confmat, class_names, "recall", fontsize=8)
        logger = self.trainer.logger
        if logger:
            logger.experiment.add_figure("Recall matrix", fig, global_step=self.trainer.global_step)
        else:
            plt.savefig('/tmp/last_conf_mat.png')

    def predict_step(self, batch, batch_idx):
        xs = batch["image"]
        logits_xs = self.forward(xs)
        return logits_xs