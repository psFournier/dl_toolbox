import pytorch_lightning as pl
import torchmetrics as M
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_
from einops import rearrange

from dl_toolbox.utils import plot_confusion_matrix


def param_groups_weight_decay(
        params,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in params:
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
            print(name)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class DecoderLinear(nn.Module):
    def __init__(self, num_classes, patch_size, d_encoder):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.head = nn.Linear(self.d_encoder, num_classes)
        self.apply(init_weights)

    def no_weight_decay(self):
        return set()

    def forward(self, x, h):
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h//self.patch_size)
        return x

class Segmenter(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        optimizer,
        scheduler,
        loss,
        batch_tf,
        metric_ignore_index,
        tta=None,
        sliding=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = timm.create_model('vit_small_patch14_dinov2', pretrained=True, dynamic_img_size=True)
        self.decoder = DecoderLinear(num_classes, 14, self.encoder.embed_dim)
        self.loss =  loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_tf = batch_tf
        self.tta = tta
        self.sliding = sliding
        metric_args = {'task':'multiclass', 'num_classes':num_classes, 'ignore_index':metric_ignore_index}
        self.val_accuracy = M.Accuracy(**metric_args)
        self.val_cm = M.ConfusionMatrix(**metric_args, normalize='true')
        self.val_jaccard = M.JaccardIndex(**metric_args)

    def configure_optimizers(self):
        train_params = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))
        print(
            f"The model will start training with only {sum([int(torch.numel(p)) for n,p in train_params])} "
            f"trainable parameters out of {sum([int(torch.numel(p)) for p in self.parameters()])}."
        )
        if hasattr(self, 'no_weight_decay'):
            wd_val = 0. #self.optimizer.weight_decay
            nwd_params = self.no_weight_decay()
            train_params = param_groups_weight_decay(train_params, wd_val, nwd_params)
            print(f"{len(train_params[0]['params'])} params do not undergo weight decay")
        optimizer = self.optimizer(params=train_params)
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }

    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))
        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params
    
    def forward(self, x, sliding=None, tta=None):
        if sliding is not None:
            auxs = [self.forward(aux, tta=tta) for aux in sliding(x)]
            return sliding.merge(auxs)
        elif tta is not None:
            auxs = [self.forward(aux) for aux in tta(x)]
            logits = self.forward(x)
            return torch.stack([logits] + self.tta.revert(auxs)).sum(dim=0)
        else:
            return self._forward(x)

    def _forward(self, im):
        H, W = im.size(2), im.size(3)
        x = self.encoder.forward_features(im)
        x = x[:,self.encoder.num_prefix_tokens:,...]
        masks = self.decoder(x, H)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        return masks
    
    def training_step(self, batch, batch_idx):
        x, y = batch["sup"]
        if self.batch_tf is not None:
            x, y = self.batch_tf(x, y)
        logits = self.forward(x)
        loss = self.loss(logits, y['masks'])
        self.log(f"{self.loss.__name__}/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x, sliding=self.sliding)
        loss = self.loss(logits, y['masks'])
        self.log(f"{self.loss.__name__}/val", loss)
        probs = self.loss.prob(logits)
        _, preds = self.loss.pred(probs)
        self.val_accuracy.update(preds, y['masks'])
        self.val_cm.update(preds, y['masks'])
        self.val_jaccard.update(preds, y['masks'])
        
    def on_validation_epoch_end(self):
        self.log("accuracy/val", self.val_accuracy.compute())
        self.log("iou/val", self.val_jaccard.compute())
        confmat = self.val_cm.compute().detach().cpu()
        self.val_accuracy.reset()
        self.val_jaccard.reset()
        self.val_cm.reset()
        class_names = self.trainer.datamodule.class_names
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        logger.experiment.add_figure("confmat/val", fig, global_step=self.trainer.global_step)