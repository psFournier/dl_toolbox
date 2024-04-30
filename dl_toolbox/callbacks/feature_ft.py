from pytorch_lightning.callbacks import BaseFinetuning
            
class FeatureFt(BaseFinetuning):
    def __init__(self, module_name, unfreeze_at_epoch=2, train_bn=True, initial_denom_lr=10.):
        super().__init__()
        self.module_name = module_name
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.train_bn = train_bn
        self.initial_denom_lr=initial_denom_lr

    def freeze_before_training(self, pl_module):
        self.freeze(getattr(pl_module, self.module_name), train_bn=self.train_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=getattr(pl_module, self.module_name),
                optimizer=optimizer,
                lr=None,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=self.train_bn,
            )