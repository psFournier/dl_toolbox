import torch
import torch.nn as nn
from dl_toolbox.lightning_modules import BaseModule
from dl_toolbox.utils import TorchOneHot
import dl_toolbox.augmentations as aug


class BCE(BaseModule):

    # BCE_multilabel = Binary Cross Entropy for multilabel prediction

    def __init__(self,
                 network,
                 weights,
                 no_pred_zero,
                 mixup,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        net_cls = self.net_factory.create(network)
        self.network = net_cls(*args, **kwargs)
        self.no_pred_zero = no_pred_zero
        self.num_classes = self.network.out_channels + int(self.no_pred_zero)
        out_dim = self.network.out_dim
        weights = torch.Tensor(weights).reshape(1, -1, *out_dim) if len(weights)>0 else None
        self.ignore_index = -1
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weights)
        self.onehot = TorchOneHot(
            range(int(self.no_pred_zero), self.num_classes)
        )
        self.mixup = aug.Mixup(alpha=mixup) if mixup > 0. else None
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--network", type=str)
        parser.add_argument("--weights", type=float, nargs="+", default=())
        parser.add_argument("--no_pred_zero", action='store_true')
        parser.add_argument("--mixup", type=float, default=0.)

        return parser

    def forward(self, x):

        return self.network(x)

    def _compute_probas(self, logits):

        return torch.sigmoid(logits)

    def _compute_conf_preds(self, probas):

        aux_confs, aux_preds = torch.max(probas, axis=1)
        if self.no_pred_zero:
            cond = aux_confs > 0.5
            preds = torch.where(cond, aux_preds + 1, 0)
            confs = torch.where(cond, aux_confs, 1-aux_confs)
            return confs, preds
        return aux_confs, aux_preds

    def training_step(self, batch, batch_idx):
        
        batch = batch["sup"]
        inputs = batch['image']
        labels = batch['mask']
        labels = self.onehot(labels).float() # B,C or C-1,H,W
        if self.mixup:
            inputs, labels = self.mixup(inputs, onehot_labels)
        logits = self.network(inputs) # B,C or C-1,H,W
        loss = self.loss(logits, labels)
        self.log('Train_sup_BCE', loss)
        batch['logits'] = logits.detach()

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        labels = batch['mask']
        logits = outs['logits']
        onehot_labels = self.onehot(labels).float() # B,C or C-1,H,W
        loss = self.loss(logits, onehot_labels)
        self.log('Val_BCE', loss)

        return outs
