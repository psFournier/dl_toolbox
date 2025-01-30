import pytorch_lightning as pl
import torchmetrics as M
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange

from dl_toolbox.utils import plot_confusion_matrix
from dl_toolbox.modules import FeatureExtractor
import torchvision.transforms.v2 as v2

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

class SegmenterLinear(pl.LightningModule):
    def __init__(
        self,
        encoder,
        class_list,
        cutmix,
        optimizer,
        scheduler,
        metric_ignore_index,
        tta=None,
        sliding=None,
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
        self.decoder = DecoderLinear(
            self.num_classes,
            self.patch_size,
            self.embed_dim
        )
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            reduction='mean',
            weight=None,
            label_smoothing=0.
        )
        self.cutmix = None
        if cutmix is not None:
            self.cutmix = v2.CutMix(
                num_classes=self.num_classes,
                alpha=cutmix
            )
        self.logits_to_probas = nn.Softmax(dim=1)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tta = tta
        self.sliding = sliding
        self.metric_args = {
            'task':'multiclass',
            'num_classes':self.num_classes,
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
        ##https://stats.stackexchange.com/questions/576463/why-not-perform-weight-decay-on-layernorm-embedding
        #if hasattr(self, 'no_weight_decay'):
        #    wd_val = 0. #weight_decay value for params that undergo it, right now no decay
        #    nwd_params = self.no_weight_decay()
        #    train_params = param_groups_weight_decay(train_params, wd_val, nwd_params)
        #    print(f"{len(train_params[0]['params'])} are not affected by weight decay.")
            
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
            features = self.feature_extractor(x)
            # We ignore non-patch tokens when recovering a segmentation by decoding
            features = features[:,self.num_prefix_tokens:,...]
            H, W = x.size(2), x.size(3)
            masks = self.decoder(features, H)
            masks = F.interpolate(masks, size=(H, W), mode="bilinear")
            return masks
    
    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        x = batch["image"]
        y = batch["target"]
        if self.cutmix is not None:
            x, y = self.cutmix(x, y)
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
        self.val_jaccard.update(preds, y)
        
    def on_validation_epoch_end(self):
        self.log("Accuracy/val", self.val_accuracy.compute())
        self.log("IoU/val", self.val_jaccard.compute())
        confmat = self.val_cm.compute().detach().cpu()
        self.val_accuracy.reset()
        self.val_jaccard.reset()
        self.val_cm.reset()
        class_names = [l.name for l in self.class_list]
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        logger.experiment.add_figure("Confusion Matrix/val", fig, global_step=self.trainer.global_step)
        
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        logits_x = self.forward(x, sliding=self.sliding)                    
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
        self.log("IoU/test", self.test_jaccard.compute())
        confmat = self.test_cm.compute().detach().cpu()
        self.test_accuracy.reset()
        self.test_jaccard.reset()
        self.test_cm.reset()
        class_names = [l.name for l in self.class_list]
        logger = self.trainer.logger
        fs = 12 - 2*(self.num_classes//10)
        fig = plot_confusion_matrix(confmat, class_names, norm=None, fontsize=fs)
        logger.experiment.add_figure("Confusion Matrix/test", fig, global_step=self.trainer.global_step)
        
    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        logits_x = self.forward(x, sliding=self.sliding)  
        probs = self.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        return probs