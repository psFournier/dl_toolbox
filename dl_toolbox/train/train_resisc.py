from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint,StochasticWeightAveraging
from pytorch_lightning import Trainer 
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from dl_toolbox.lightning_modules import *
from dl_toolbox.lightning_datamodules import *
from dl_toolbox.callbacks import *
from dl_toolbox.torch_datasets import *
from dl_toolbox.utils import LabelsToRGB

models = {
    'Smp_Unet_CE': {
        'cls': Smp_Unet_CE,
        'datamodule_cls': SupervisedDm
    },
    'Unet_CE': {
        'cls': Unet_CE,
        'datamodule_cls': SupervisedDm
    },
    'Smp_Unet_BCE_binary': {
        'cls': Smp_Unet_BCE_binary,
        'datamodule_cls': SupervisedDm
    },
    'Smp_Unet_BCE_multilabel': {
        'cls': Smp_Unet_BCE_multilabel,
        'datamodule_cls': SupervisedDm
    },   
    'Smp_Unet_BCE_multilabel_2': {
        'cls': Smp_Unet_BCE_multilabel_2,
        'datamodule_cls': SupervisedDm
    },
    'Smp_Unet_BCE_Mixup': {
        'cls': Smp_Unet_BCE_Mixup,
        'datamodule_cls': SupervisedDm
    },
    'PL': {
        'cls': PL,
        'datamodule_cls': SemisupDm
    },
    'CPS': {
        'cls': CPS,
        'datamodule_cls': SemisupDm
    },
    'MT': {
        'cls': MT,
        'datamodule_cls': SemisupDm
    },
    'PLM': {
        'cls': PLM,
        'datamodule_cls': SemisupDm
    }
}

def main():

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--model", type=str)

    parser = ResiscDm.add_model_specific_args(parser)

    args = parser.parse_known_args()[0]
    
    parser = models[args.model]['cls'].add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args_dict = vars(args)

    pl_datamodule = ResiscDm(**args_dict)
    pl_module = models[args.model]['cls'](**args_dict)
    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger(args.output_dir, version=args.version, name=args.exp_name),
        profiler=SimpleProfiler(),
        callbacks=[
            ModelCheckpoint(),
            ConfMatLogger(
                labels=['forest', 'harbor'],
                freq=1
            ),
            CalibrationLogger(freq=1),
            ClassDistribLogger(freq=1)
            #StochasticWeightAveraging(
            #    swa_epoch_start=0.91,
            #    swa_lrs=0.005,
            #    annealing_epochs=1,
            #    annealing_strategy='linear',
            #    device=None
            #),
        ],
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        benchmark=True
    )

    trainer.fit(model=pl_module, datamodule=pl_datamodule)


if __name__ == "__main__":

    main()
