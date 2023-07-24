import os
import time
from argparse import ArgumentParser

import segmentation_models_pytorch as smp
import tabulate
import torch
from dl_toolbox.lightning_datamodules import SplitfilSup
from dl_toolbox.lightning_modules import CE
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import SemcityBdsdDs
from dl_toolbox.utils import worker_init_function
from rasterio.windows import Window
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)

    parser = SplitfileSup.add_model_specific_args(parser)
    parser = CE.add_model_specific_args(parser)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--train_with_void", action="store_true")
    parser.add_argument("--eval_with_void", action="store_true")
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--initial_lr", type=float)
    parser.add_argument("--final_lr", type=float)
    parser.add_argument("--lr_milestones", nargs=2, type=float)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--img_aug", type=str, default="no")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    datamodule = SplitfileSup(**args)

    dataset1 = SemcityBdsdDs(
        image_path=os.path.join(args.data_path, "BDSD_M_3_4_7_8.tif"),
        label_path=os.path.join(args.data_path, "GT_3_4_7_8.tif"),
        fixed_crops=False,
        tile=Window(col_off=0, row_off=0, width=2000, height=2000),
        crop_size=args.crop_size,
        crop_step=args.crop_size,
        img_aug=args.img_aug,
        labels="building",
    )
    dataset2 = SemcityBdsdDs(
        image_path=os.path.join(args.data_path, "BDSD_M_3_4_7_8.tif"),
        label_path=os.path.join(args.data_path, "GT_3_4_7_8.tif"),
        fixed_crops=False,
        tile=Window(col_off=2000, row_off=2000, width=2000, height=2000),
        crop_size=args.crop_size,
        crop_step=args.crop_size,
        img_aug=args.img_aug,
        labels="building",
    )
    trainset = ConcatDataset([dataset1, dataset2])

    train_sampler = RandomSampler(
        data_source=trainset, replacement=True, num_samples=args.epoch_len
    )

    train_dataloader = DataLoader(
        dataset=trainset,
        batch_size=args.sup_batch_size,
        collate_fn=CustomCollate(batch_aug="none"),
        drop_last=True,
        worker_init_fn=worker_init_function,
        sampler=train_sampler,
        num_workers=args.workers,
    )

    valset = SemcityBdsdDs(
        image_path=os.path.join(args.data_path, "BDSD_M_3_4_7_8.tif"),
        label_path=os.path.join(args.data_path, "GT_3_4_7_8.tif"),
        fixed_crops=True,
        tile=Window(col_off=4000, row_off=4000, width=2000, height=2000),
        crop_size=args.crop_size,
        crop_step=args.crop_size,
        img_aug="no",
        labels="building",
    )

    val_dataloader = DataLoader(
        dataset=valset,
        shuffle=False,
        batch_size=args.sup_batch_size,
        num_workers=args.workers,
        worker_init_fn=worker_init_function,
    )

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_use_batchnorm=True,
    )

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    columns = ["ep", "train_loss", "val_loss", "time"]

    for epoch in range(start_epoch, args.max_epochs):
        time_ep = time.time()
        loss_sum = 0.0
        model.train()

        for _, batch in enumerate(train_dataloader):
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_res = {"loss": loss_sum / len(train_dataloader)}
        loss_sum = 0.0
        model.eval()

        for _, batch in enumerate(val_dataloader):
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            output = model(image)
            loss = loss_fn(output, target)
            loss_sum += loss.item()

        val_res = {"loss": loss_sum / len(val_dataloader)}
        time_ep = time.time() - time_ep
        values = [epoch + 1, train_res["loss"], val_res["loss"], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        print(table)


if __name__ == "__main__":
    main()
