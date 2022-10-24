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
    parser.add_argument("--datamodule", type=str)
    parser.add_argument("--model", type=str)

    args = parser.parse_known_args()[0]

    datamodule_cls = DatamoduleFactory().create(args.datamodule)
    parser = datamodule_cls.add_model_specific_args(parser)

    module_cls = ModuleFactory().create(args.model)
    parser = module_cls.add_model_specific_args(parser)

    args = parser.parse_known_args()[0]

    network_cls = NetworkFactory().create(args.network)
    parser = network_cls.add_model_specific_args(parser)

    try:
        dataset_cls = DatasetFactory().create(args.dataset)
        parser = dataset_cls.add_model_specific_args(parser)
    except:
        pass

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args_dict = vars(args)

    #dataset = dataset_cls(**args_dict)
    datamodule = datamodule_cls(**args_dict)
    module = module_cls(**args_dict)

    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger(args.output_dir, version=args.version, name=args.exp_name),
        #profiler=SimpleProfiler(),
        callbacks=[
            ModelCheckpoint(),
            #DeviceStatsMonitor(),
        ],
        log_every_n_steps=100,
        flush_logs_every_n_steps=100,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        benchmark=True
    )

    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":

    main()
