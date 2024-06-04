from pytorch_lightning.callbacks import BaseFinetuning
            
class Finetuning(BaseFinetuning):
    def __init__(self, module_name, unfreeze_at_epoch, train_bn, initial_denom_lr, activated):
        super().__init__()
        self.module_name = module_name
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.train_bn = train_bn
        self.initial_denom_lr=initial_denom_lr
        self.activated = activated

    def freeze_before_training(self, pl_module):
        if self.activated:
            self.freeze(getattr(pl_module, self.module_name), train_bn=self.train_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if self.activated and current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=getattr(pl_module, self.module_name),
                optimizer=optimizer,
                lr=None,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=self.train_bn,
            )