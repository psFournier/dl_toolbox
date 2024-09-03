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
    def __init__(self, num_classes, patch_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.head = nn.Linear(self.embed_dim, num_classes)
        # self.apply applies to every submodule recursively, typical for init
        self.apply(init_weights)

    def no_weight_decay(self):
        return set()

    def forward(self, x, img_height):
        x = self.head(x) # BxNpatchxEmbedDim -> BxNpatchxNcls
        num_patch_h = img_height//self.patch_size # num_patch_per_axis Np
        # rearrange auto computes h & w from the fixed val h; C-order enum used, see docs
        x = rearrange(x, "b (h w) c -> b c h w", h=num_patch_h) # BxNclsxNpxNp
        return x

class Segmenter(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        backbone,
        optimizer,
        scheduler,
        onehot,
        loss,
        metric_ignore_index,
        tta=None,
        sliding=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone, # Name of the pretrained model
            pretrained=True,
            dynamic_img_size=True # Deals with image sizes != from pretraining
        )
        self.decoder = DecoderLinear(
            num_classes,
            self.backbone.patch_embed.patch_size[0], # patch_size
            self.backbone.embed_dim
        )
        self.loss =  loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.onehot = onehot
        self.tta = tta
        self.sliding = sliding
        self.metric_args = {
            'task':'multiclass',
            'num_classes':num_classes,
            'ignore_index':metric_ignore_index # should metrics (not loss) ignore an index
        }
        self.val_accuracy = M.Accuracy(**self.metric_args)
        self.val_cm = M.ConfusionMatrix(**self.metric_args, normalize='true')
        self.val_jaccard = M.JaccardIndex(**self.metric_args)

    def configure_optimizers(self):
        train_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        nb_train = sum([int(torch.numel(p)) for p in train_params])
        nb_tot = sum([int(torch.numel(p)) for p in self.parameters()])
        print(f"Training {nb_train} params out of {nb_tot}.")
        
        ##https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025
        #if hasattr(self, 'no_weight_decay'):
        #    wd_val = 0. #weight_decay value for params that undergo it, right now no decay
        #    nwd_params = self.no_weight_decay()
        #    train_params = param_groups_weight_decay(train_params, wd_val, nwd_params)
        #    print(f"{len(train_params[0]['params'])} are not affected by weight decay.")
            
        optimizer = self.optimizer(params=train_params)
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

    def no_weight_decay(self):
        """The names in this module of the parameters we should remove weight decay from must be taken from their names in the timm lib or in the decoder
        """
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))
        nwd_params = append_prefix_no_weight_decay("backbone.", self.backbone).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params
    
    def forward(self, x, sliding=None, tta=None):
        # First we subdivide the image in windows, forward on each, and merge
        if sliding is not None:
            auxs = [self.forward(aux, tta=tta) for aux in sliding(x)]
            return sliding.merge(auxs)
        # For each window, we apply norma+tta and merge
        elif tta is not None:
            auxs = [self.forward(aux) for aux in tta(x)]
            logits = self.forward(x)
            return torch.stack([logits] + self.tta.revert(auxs)).sum(dim=0)
        else:
            # Base forward
            return self._forward(x)

    def _forward(self, im):
        H, W = im.size(2), im.size(3)
        x = self.backbone.forward_features(im)
        # We ignore non-patch tokens when recovering a segmentation by decoding
        x = x[:,self.backbone.num_prefix_tokens:,...]
        masks = self.decoder(x, H)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        return masks
    
    def _onehot(self, y):
        return torch.movedim(F.one_hot(y, self.num_classes),-1,1).float()
    
    def training_step(self, batch, batch_idx):
        x, tgt, p = batch["sup"]
        y = tgt['masks']
        if self.onehot: y = self._onehot(y)
        logits = self.forward(x) #without sliding nor tta
        loss = self.loss(logits, y)
        self.log("Total loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, tgt, p = batch
        y = tgt['masks']
        logits = self.forward(x) # no slide, no tta
        loss = self.loss(logits, y)
        self.log("Total loss/val", loss)
        probs = self.loss.prob(logits)
        _, preds = self.loss.pred(probs)
        self.val_accuracy.update(preds, y)
        self.val_cm.update(preds, y)
        self.val_jaccard.update(preds, y)
        
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
        
    def predict_step(self, batch, batch_idx):
        x, y, p = batch
        logits = self.forward(x, sliding=self.sliding, tta=self.tta)
        return self.loss.prob(logits)