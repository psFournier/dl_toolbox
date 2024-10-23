from pytorch_lightning.callbacks import BaseFinetuning
import torch
            
class Finetuning(BaseFinetuning):
    def __init__(self, module_name, unfreeze_at, train_bn, initial_denom_lr, activated):
        super().__init__()
        self.module_name = module_name
        self.unfreeze_at = unfreeze_at # here, step to unfreeze at
        self.train_bn = train_bn
        self.initial_denom_lr=initial_denom_lr
        self.activated = activated

    def freeze_before_training(self, pl_module):
        if self.activated:
            self.freeze(getattr(pl_module, self.module_name), train_bn=self.train_bn)
    
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the epoch begins."""
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            num_param_groups = len(optimizer.param_groups)
            self.finetune_function(pl_module, trainer.current_epoch, optimizer)
            #self.finetune_function(pl_module, trainer.global_step, optimizer)
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def finetune_function(self, pl_module, current, optimizer):
        if self.activated and current == self.unfreeze_at:
            self.unfreeze_and_add_param_group(
                modules=getattr(pl_module, self.module_name),
                optimizer=optimizer,
                lr=None,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=self.train_bn,
            )
        for i, p in enumerate(optimizer.param_groups):
            lr = p['lr']
            print(f'epoch {current}: param group {i} lr={lr}')
            
    @staticmethod
    def unfreeze_and_add_param_group(
        modules,
        optimizer,
        lr=None,
        initial_denom_lr=10.0,
        train_bn=True,
    ):
        """Unfreezes a module and adds its parameters to an optimizer.

        Args:
            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.
            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`
            lr: Learning rate for the new param group.
            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by `initial_denom_lr`.
            train_bn: Whether to train the BatchNormalization layers.

        """
        #starting_params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        #print(f'At start, {sum([int(torch.numel(p)) for p in starting_params])} trainable params')
        BaseFinetuning.make_trainable(modules)
        params = list(BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True))
        print(f'After adding all, {sum([int(torch.numel(p)) for p in params])} trainable params')
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        print(f'Then after filtering on optimizer, {sum([int(torch.numel(p)) for p in params])} trainable params')
        params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.0
        if params:
            #optimizer.add_param_group({"params": params, "lr": 0.})
            optimizer.add_param_group({"params": params, "lr": params_lr / denom_lr})
            
