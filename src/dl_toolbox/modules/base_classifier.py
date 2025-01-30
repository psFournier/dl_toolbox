import pytorch_lightning as pl
import torch
import torchmetrics as M
from dl_toolbox.utils import plot_confusion_matrix
from pytorch_lightning.utilities import rank_zero_info
import torch.nn as nn
from timm.models.layers import trunc_normal_

from dl_toolbox.modules import FeatureExtractor

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class DecoderLinear(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.apply(init_weights)

    def no_weight_decay(self):
        return set()

    def forward(self, x):
        #x = x[:,1:]
        #avg = x.mean(dim=1)
        #norm = self.fc_norm(avg)
        x = x[:,0]
        logits = self.head(x)
        return logits


class BaseClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder,
        class_list,
        optimizer,
        scheduler,
        tta=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)
        self.feature_extractor = FeatureExtractor(encoder)
        self.num_prefix_tokens = self.feature_extractor.encoder.num_prefix_tokens
        self.embed_dim = self.feature_extractor.encoder.embed_dim
        self.patch_size = self.feature_extractor.encoder.patch_embed.patch_size[0]
        self.decoder = DecoderLinear(self.num_classes, self.embed_dim)
        #self.model = timm.create_model(
        #    encoder,
        #    pretrained=True,
        #    num_classes=self.num_classes
        #)
        #self.feature_extractor = create_feature_extractor(
        #    self.model,
        #    {'conv_head': 'features'}
        #)
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            reduction='mean',
            weight=None,
            label_smoothing=0.
        )
        self.logits_to_probas = nn.Softmax(dim=1)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tta = tta
        metric_args = {
            'task': 'multiclass',
            'num_classes': self.num_classes
        }
        self.val_accuracy = M.Accuracy(**metric_args)
        self.test_accuracy = M.Accuracy(**metric_args)
        self.val_cm = M.ConfusionMatrix(**metric_args, normalize='true')
        self.test_cm = M.ConfusionMatrix(**metric_args, normalize='true')
    
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {sum([int(torch.numel(p)) for p in trainable_parameters])} "
            f"trainable parameters out of {sum([int(torch.numel(p)) for p in parameters])}."
        )
        optimizer = self.optimizer(params=trainable_parameters)
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"#"step"
            },
        }

    def forward(self, x, tta=None):
        if tta:
            auxs = [self.forward(aux) for aux in tta(x)]
            logits = self.forward(x)
            return torch.stack([logits] + self.tta.revert(auxs)).sum(dim=0)
        else:
            features = self.feature_extractor(x)
            return self.decoder(features)
            #return self.model.forward(x)
            
    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        x = batch["image"]
        y = batch["target"]
        logits_x = self.forward(x)
        loss = self.loss(logits_x, y)
        self.log(f"Cross Entropy/train", loss)
        self.log(f"Loss/train", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        logits_x = self.forward(x)                    
        loss = self.loss(logits_x, y)
        self.log(f"Cross Entropy/val", loss)
        self.log(f"Loss/val", loss)
        probs = self.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        self.val_accuracy.update(preds, y)
        self.val_cm.update(preds, y)
        
    def on_validation_epoch_end(self):
        self.log("Accuracy/val", self.val_accuracy.compute())
        confmat = self.val_cm.compute().detach().cpu()
        self.val_accuracy.reset()
        self.val_cm.reset()
        class_names = [l.name for l in self.class_list]
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        logger.experiment.add_figure("Confusion Matrix/val", fig, global_step=self.trainer.global_step)

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        logits_x = self.forward(x, tta=self.tta)
        loss = self.loss(logits_x, y)
        self.log(f"Cross Entropy/test", loss)
        self.log(f"Loss/test", loss)
        probs = self.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        self.test_accuracy.update(preds, y)
        self.test_cm.update(preds, y)
        self.test_jaccard.update(preds, y)
        
    def on_test_epoch_end(self):
        self.log("Accuracy/test", self.test_accuracy.compute())
        confmat = self.test_cm.compute().detach().cpu()
        self.test_accuracy.reset()
        self.test_cm.reset()
        class_names = [l.name for l in self.class_list]
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        logger.experiment.add_figure("Confusion Matrix/test", fig, global_step=self.trainer.global_step)

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        logits_x = self.forward(x, tta=self.tta)
        probs = self.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        return probs