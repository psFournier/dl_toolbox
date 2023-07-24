import torch
import torch.nn as nn

from dl_toolbox.lightning_modules import BCE
import dl_toolbox.utils as utils
from dl_toolbox.torch_datasets.utils import get_transforms


class BCE_PL(BCE):
    def __init__(
        self,
        final_alpha,
        alpha_milestones,
        pseudo_threshold,
        consist_aug,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.final_alpha = final_alpha
        self.alpha_milestones = alpha_milestones
        self.pseudo_threshold = pseudo_threshold
        self.consist_aug = get_transforms(consist_aug)
        # The BCE loss must be used because the network has been trained to
        # produce preds such that applying a sigmoid gives probas ; could use
        # logsigmoid followed by NLL loss instead to keep CE
        self.pl_loss = nn.BCEWithLogitsLoss(reduction="none")

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--final_alpha", type=float)
        parser.add_argument("--alpha_milestones", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--consist_aug", type=str)

        return parser

    def on_train_epoch_start(self):
        self.alpha = utils.ramp_down(
            self.trainer.current_epoch, *self.alpha_milestones, self.final_alpha
        )

    def training_step(self, batch, batch_idx):
        outs = super().training_step(batch, batch_idx)
        batch, unsup_batch = batch["sup"], batch["unsup"]
        self.log("Prop unsup train", self.alpha)

        if self.alpha > 0:
            unsup_inputs = unsup_batch["image"]
            pl_logits = self.network(unsup_inputs).detach()
            pl_probas = self._compute_probas(pl_logits)
            aug_unsup_inputs, aug_pl_probas = self.consist_aug(
                img=unsup_inputs, label=pl_probas
            )
            pl_confs, pl_preds = self._compute_conf_preds(aug_pl_probas)
            pl_certain = (pl_confs > self.pseudo_threshold).float()
            pl_certain_sum = torch.sum(pl_certain) * self.network.out_channels + 1e-5
            self.log("PL certainty prop", torch.mean(pl_certain))
            aug_pl_logits = self.network(aug_unsup_inputs)
            onehot_pl_labels = self.onehot(pl_preds).float()
            pl_loss = self.pl_loss(aug_pl_logits, onehot_pl_labels)
            pl_loss = torch.sum(pl_certain * pl_loss) / pl_certain_sum
            self.log("PL loss", pl_loss)
            outs["loss"] += self.alpha * pl_loss
            if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
                log_batch_images(unsup_batch, self.trainer, prefix="Unsup_train")

        return outs

    def validation_step(self, batch, batch_idx):
        outs = super().validation_step(batch, batch_idx)
        logits = outs["logits"]
        probas = self._compute_probas(logits.detach())
        confs, preds = self._compute_conf_preds(probas)
        certain = (confs > self.pseudo_threshold).float()
        certain_sum = torch.sum(certain) + 1e-5
        self.log("Val certainty prop", torch.mean(certain))
        acc = preds.eq(batch["mask"]).float()
        pl_acc = torch.sum(certain * acc) / certain_sum
        self.log("Val acc of pseudo labels", pl_acc)

        return outs
