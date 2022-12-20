import torch
import torch.nn as nn
from copy import deepcopy

from dl_toolbox.lightning_modules import CE
import dl_toolbox.utils as utils
from dl_toolbox.torch_datasets.utils import get_transforms
from dl_toolbox.callbacks import log_batch_images


class CE_MT(CE):

    def __init__(
        self,
        alphas,
        ramp,
        pseudo_threshold,
        consist_aug,
        emas,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_network = deepcopy(self.network)
        self.alphas = alphas
        self.ramp = ramp
        self.pseudo_threshold = pseudo_threshold
        self.consist_aug = get_transforms(consist_aug)
        self.emas = emas
        self.pl_loss = nn.CrossEntropyLoss(
            reduction='none'
        )
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--emas", nargs=2, type=float)
        parser.add_argument("--alphas", nargs=2, type=float)
        parser.add_argument("--ramp", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--consist_aug", type=str)

        return parser

    def on_train_epoch_start(self):
        
        self.alpha = utils.sigm_ramp(
            self.trainer.global_step,
            *self.ramp,
            *self.alphas
        )

        self.ema = utils.sigm_ramp(
            self.trainer.global_step,
            *self.ramp,
            *self.emas
        )

    def training_step(self, batch, batch_idx):
        
        outs = super().training_step(batch, batch_idx)
        batch, unsup_batch = batch["sup"], batch["unsup"]
        self.log('Prop unsup train', self.alpha)
        
        if self.alpha > 0:
            
            unsup_inputs = unsup_batch['image']
            with torch.no_grad():
                pl_logits = self.teacher_network(unsup_inputs)
            pl_probas = self._compute_probas(pl_logits)
            aug_unsup_inputs, aug_pl_probas = self.consist_aug(
                img=unsup_inputs,
                label=pl_probas
            )
            pl_confs, pl_preds = self._compute_conf_preds(aug_pl_probas)
            pl_certain = (pl_confs > self.pseudo_threshold).float()
            pl_certain_sum = torch.sum(pl_certain) + 1e-5
            self.log('PL certainty prop', torch.mean(pl_certain))
            aug_pl_logits = self.network(aug_unsup_inputs)
            pl_loss = self.pl_loss(aug_pl_logits, pl_preds)
            pl_loss = torch.sum(pl_certain * pl_loss) / pl_certain_sum
            self.log('PL loss', pl_loss)
            outs['loss'] += self.alpha * pl_loss
            if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
                log_batch_images(
                    unsup_batch,
                    self.trainer,
                    prefix='Unsup_train'
                )

        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

        return outs

    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        logits = outs['logits']
        probas = self._compute_probas(logits.detach())
        confs, preds = self._compute_conf_preds(probas)
        certain = (confs > self.pseudo_threshold).float()
        certain_sum = torch.sum(certain) + 1e-5
        self.log('Val certainty prop', torch.mean(certain))
        acc = preds.eq(batch['mask']).float()
        pl_acc = torch.sum(certain * acc) / certain_sum
        self.log('Val acc of pseudo labels', pl_acc)

        return outs


