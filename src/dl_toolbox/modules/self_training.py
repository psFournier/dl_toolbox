import torch
from .supervised import Supervised


class SelfTraining(Supervised):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        batch, pseudosup_batch = batch["sup"], batch["pseudosup"]
        xs = batch["image"]
        ys = batch["label"]
        # ST    
        x_pl = pseudosup_batch["image"]
        y_pl = pseudosup_batch["label"]
        # strong augment on pseudosup batch ?
        x_all = torch.vstack((xs, x_pl))
        y_all = torch.vstack((ys, y_pl))
        x, y = self.apply_batch_tf(x_all, y_all)
        logits_x = self.forward(x)
        ce = self.ce_train(logits_x, y)
        self.log(f"cross_entropy/train", ce)
        dice = 0
        #if self.dice is not None: 
        #    dice = self.dice(logits_x, self.one_hot(torch.vstack((ys, y_pl))))
        #    self.log(f"dice/train", dice)
        return ce + dice