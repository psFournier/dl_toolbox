import gc
import os

import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from os.path import join

from featup.datasets.JitteredImage import apply_jitter, sample_transform
from featup.datasets.util import get_dataset, SingleImageDataset
from featup.downsamplers import SimpleDownsampler, AttentionDownsampler
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.losses import TVLoss, SampledCRFLoss, entropy
from featup.upsamplers import get_upsampler
from featup.util import pca, RollingAvg, unnorm, norm, prep_image

class JBUFeatUp(pl.LightningModule):
    def __init__(self,
                 model_type,
                 activation_type,
                 n_jitters,
                 max_pad,
                 max_zoom,
                 kernel_size,
                 final_size,
                 lr,
                 random_projection,
                 upsampler,
                 chkpt_dir
                 ):
        super().__init__()
        self.model_type = model_type
        self.activation_type = activation_type
        self.n_jitters = n_jitters
        self.max_pad = max_pad
        self.max_zoom = max_zoom
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.lr = lr
        self.random_projection = random_projection
        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = torch.nn.Sequential(self.model, ChannelNorm(self.dim))
        self.upsampler = get_upsampler(upsampler, self.dim)
        self.downsampler = SimpleDownsampler(self.kernel_size, self.final_size)
        self.avg = RollingAvg(20)
        self.automatic_optimization = False
        self.chkpt_dir = chkpt_dir
        
    def forward(self, x):
        return self.upsampler(self.model(x))

    def project(self, feats, proj):
        if proj is None:
            return feats
        else:
            return torch.einsum("bchw,bcd->bdhw", feats, proj)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if type(batch) == dict:
                img = batch['image']
            else:
                img, _ = batch
            lr_feats = self.model(img)

        full_rec_loss = 0.0
        for i in range(self.n_jitters):
            hr_feats = self.upsampler(lr_feats, img)

            if hr_feats.shape[2] != img.shape[2]:
                hr_feats = torch.nn.functional.interpolate(hr_feats, img.shape[2:], mode="bilinear")

            with torch.no_grad():
                transform_params = sample_transform(
                    True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3])
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

            if self.random_projection is not None:
                proj = torch.randn(lr_feats.shape[0],
                                   lr_feats.shape[1],
                                   self.random_projection, device=lr_feats.device)
                proj /= proj.square().sum(1, keepdim=True).sqrt()
            else:
                proj = None

            hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
            proj_hr_feats = self.project(hr_jit_feats, proj)

            down_jit_feats = self.project(self.downsampler(hr_jit_feats, jit_img), proj)

            rec_loss = (self.project(lr_jit_feats, proj) - down_jit_feats).square().mean() / self.n_jitters

            full_rec_loss += rec_loss.item()

            self.manual_backward(rec_loss)
            
        self.avg.add("loss/rec", full_rec_loss)
        self.avg.logall(self.log)
        if self.global_step % 100 == 0:
            self.trainer.save_checkpoint(self.chkpt_dir[:-5] + '/' + self.chkpt_dir[:-5] + f'_{self.global_step}.ckpt')

        if self.global_step < 10:
            self.clip_gradients(opt, gradient_clip_val=.0001, gradient_clip_algorithm="norm")

        opt.step()

        return None
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.trainer.is_global_zero and batch_idx == 0:

                if type(batch) == dict:
                    img = batch['image']
                else:
                    img, _ = batch
                lr_feats = self.model(img)

                hr_feats = self.upsampler(lr_feats, img)

                if hr_feats.shape[2] != img.shape[2]:
                    hr_feats = torch.nn.functional.interpolate(hr_feats, img.shape[2:], mode="bilinear")

                transform_params = sample_transform(
                    True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3])
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

                if self.random_projection is not None:
                    proj = torch.randn(lr_feats.shape[0],
                                       lr_feats.shape[1],
                                       self.random_projection, device=lr_feats.device)
                    proj /= proj.square().sum(1, keepdim=True).sqrt()
                else:
                    proj = None
                    
                #scales = self.scale_net(lr_jit_feats)

                writer = self.logger.experiment

                hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
                down_jit_feats = self.downsampler(hr_jit_feats, jit_img)
                down_jit_feats_proj = self.project(down_jit_feats, proj)
                
                rec_loss = (self.project(lr_jit_feats, proj) - down_jit_feats_proj).square().mean()
                print(rec_loss.item())


                [red_lr_feats], fit_pca = pca([lr_feats[0].unsqueeze(0)])
                [red_hr_feats], _ = pca([hr_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_lr_jit_feats], _ = pca([lr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_hr_jit_feats], _ = pca([hr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_down_jit_feats], _ = pca([down_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)

                writer.add_image("viz/image", unnorm(img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("viz/lr_feats", red_lr_feats[0], self.global_step)
                writer.add_image("viz/hr_feats", red_hr_feats[0], self.global_step)
                writer.add_image("jit_viz/jit_image", unnorm(jit_img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("jit_viz/lr_jit_feats", red_lr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/hr_jit_feats", red_hr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/down_jit_feats", red_down_jit_feats[0], self.global_step)
#
                #norm_scales = scales[0]
                #norm_scales /= scales.max()
                #writer.add_image("scales", norm_scales, self.global_step)
                #writer.add_histogram("scales hist", scales, self.global_step)
#
                writer.add_image(
                    "down/filter",
                    prep_image(self.downsampler.get_kernel().squeeze(), subtract_min=False),
                    self.global_step)
#
                writer.flush()

    def configure_optimizers(self):
        all_params = []
        all_params.extend(list(self.downsampler.parameters()))
        all_params.extend(list(self.upsampler.parameters()))
        return torch.optim.NAdam(all_params, lr=self.lr)
    
import torchvision.transforms.v2 as v2
import dl_toolbox.datasets as datasets
from torch.utils.data import Subset, RandomSampler
import torch
from dl_toolbox.utils import CustomCollate


transform = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

NB_IMG = 45*700
dataset = datasets.Resisc('/data/NWPU-RESISC45', transform, 'all45')
trainset = Subset(dataset, indices=[i for i in range(NB_IMG) if 100<=i%700])
valset = Subset(dataset, indices=[i for i in range(NB_IMG) if 100>i%700])

train_loader = torch.utils.data.DataLoader(
    trainset,
    collate_fn=CustomCollate(),
    num_workers=6,
    pin_memory=True,
    sampler=RandomSampler(
        trainset,
        replacement=True,
        num_samples=1000
    ),
    drop_last=True,
    batch_size=1,
)
val_loader = torch.utils.data.DataLoader(
    Subset(trainset, indices=[0]),
    collate_fn=CustomCollate(),
    num_workers=6,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    batch_size=1,
)

from dl_toolbox.callbacks import ProgressBar

log_dir = '/data/outputs/jbu'
chkpt_dir = '/data/outputs/jbu/test.ckpt'
os.makedirs(log_dir, exist_ok=True)
tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
callbacks = [ModelCheckpoint(chkpt_dir[:-5], every_n_epochs=1)]

module = JBUFeatUp(
    model_type="vit",
    activation_type="token",
    n_jitters=2,
    max_pad=20,
    max_zoom=2,
    kernel_size=16,
    final_size=14,
    lr=1e-3,
    random_projection=30,
    upsampler="jbu_stack",
    chkpt_dir=chkpt_dir
)

gc.collect()
torch.cuda.empty_cache()
gc.collect()
                 
trainer = pl.Trainer(
    accelerator='gpu',
    logger=tb_logger,
    devices=1,
    max_epochs=100,
    limit_train_batches=1.,
    limit_val_batches=1.,
    callbacks=callbacks+[ProgressBar()],
    val_check_interval=100,
    log_every_n_steps=10,
)

trainer.fit(
    module,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)