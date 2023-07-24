from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import SWALR

from dl_toolbox.lightning_modules import Unet

module = Unet(
    encoder="efficientnet-b0",
    pretrained=False,
    in_channels=3,
    num_classes=9,
    learning_rate=0.05,
)


optimizer = SGD(module.parameters(), lr=module.learning_rate, momentum=0.9)


def lambda_lr(epoch):
    # s = self.trainer.max_steps
    # b = self.trainer.datamodule.sup_batch_size
    # l = self.trainer.datamodule.epoch_len
    # m = s * b / l
    m = 300
    if epoch < 0.4 * m:
        return 1
    elif 0.4 * m <= epoch <= 0.7 * m:
        return 1 + ((epoch - 0.4 * m) / (0.7 * m - 0.4 * m)) * (0.01 - 1)
    else:
        return 0.01


scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
swa_start = 240
swa_scheduler = SWALR(optimizer, swa_lr=0.05)
for epoch in range(300):
    for i in range(50):
        optimizer.step()
    if epoch > swa_start:
        swa_scheduler.step()
    else:
        scheduler.step()
    for param_group in optimizer.param_groups:
        print(param_group["lr"])

# Update bn statistics for the swa_model at the end
# torch.optim.swa_utils.update_bn(loader, swa_model)
