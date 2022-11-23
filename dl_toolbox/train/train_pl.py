from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning import Trainer 
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from dl_toolbox.lightning_modules import *
from dl_toolbox.lightning_datamodules import *
from dl_toolbox.networks import *
from dl_toolbox.torch_datasets import *

def main():

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser = CE.add_model_specific_args(parser)
    parser = Vgg.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args_dict = vars(args)

    module = CE(**args_dict)

    logger = TensorBoardLogger(
        args.output_dir,
        version=args.version,
        name=args.exp_name
    )

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        #profiler=SimpleProfiler(),
        callbacks=[
            ModelCheckpoint(),
            #DeviceStatsMonitor(),
        ]
    )

    train_set = Resisc(
        idx=tuple(range(1, 50)),
        data_path='/d/pfournie/ai4geo/data/NWPU-RESISC45',
        img_aug='d4',
        labels='base'
    )
    val_set = Resisc(
        idx=tuple(range(50, 100)),
        data_path='/d/pfournie/ai4geo/data/NWPU-RESISC45',
        img_aug='no',
        labels='base'
    )

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.sup_batch_size,
        collate_fn=CustomCollate(),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_function,
        drop_last=True
    )

    val_dataloader = Dataloader(
        dataset=val_set,
        shuffle=False,
        collate_fn=CustomCollate(),
        batch_size=args.sup_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )

    trainer.fit(
        model=module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    pred_set = Resisc(
        idx=tuple(range(50, 150)),
        data_path='/d/pfournie/ai4geo/data/NWPU-RESISC45',
        labels='base',
        img_aug='no'
    )
    
    pred_dataloader = Dataloader(
        dataset=pred_set,
        shuffle=False,
        collate_fn=CustomCollate(),
        batch_size=args.sup_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )

    preds = trainer.predict(
        model=module,
        dataloaders=pred_dataloader,
        return_predictions = True
    )

    pl_dir = os.path.join(
        logger.save_dir,
        logger.name,
        logger.log_dir,
        'NWPU-RESISC45'
    )

    for pred in preds:
        pl_set.write(
            pred,
            dst=pl_dir
        )
    
    pl_set = Resisc(
        idx=tuple(range(50, 150)),
        data_path=pl_dir,
        labels='base',
        img_aug='d4_color-5'
    )

    pl_dataloader = Dataloader(
        dataset=pl_set,
        shuffle=True,
        collate_fn=CustomCollate(),
        batch_size=args.sup_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )

    semisup_dataloader = {
        'sup': train_dataloader,
        'unsup': pl_dataloader
    }

    module = CE_PLseq()



if __name__ == "__main__":

    main()
