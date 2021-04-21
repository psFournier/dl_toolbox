import datetime
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, loggers
from pl_modules import *
from pl_datamodules import *

modules = {
    'supervised_baseline': SupervisedBaseline,
    'mean_teacher': MeanTeacher
}

datamodules = {
    'isprs_vai_semisup': IsprsVaiSemisup,
    'isprs_vai_sup': IsprsVaiSup,
    'miniworld_semisup': MiniworldSemisup,
    'miniworld_sup': MiniworldSup
}

def main():

    # Reading parameters
    parser = ArgumentParser()
    parser.add_argument("--datamodule", type=str, default='isprs_vai_sup')
    parser.add_argument("--module", type=str, default="supervised_baseline")

    # Datamodule and module classes define their own specific command line
    # arguments, so we need them to go further with the parser.
    args = parser.parse_known_args()[0]
    parser = datamodules[args.datamodule].add_model_specific_args(parser)
    parser = modules[args.module].add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    args = parser.parse_args()
    args_dict = vars(args)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    tensorboard = loggers.TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard",
        version="%s_%s" % (args.exp_name, current_date)
    )

    # The lightning datamodule deals with instantiating the proper dataloaders.
    pl_datamodule = datamodules[args.datamodule](**args_dict)

    # The lightning module is where the training schema is implemented. Class
    # weights are a property of the dataset being processed, given by its class.
    args_dict['class_weights'] = datamodules[args.datamodule].class_weights
    pl_module = modules[args.module](**args_dict)

    # Callback to save the weights (not the full state) of the
    # model with maximum validation IoU.
    checkpoint_callback = ModelCheckpoint(
        monitor='val_IoU',
        mode='max',
        save_weights_only=True
    )

    # Callback to log the learning rate
    lr_monitor = LearningRateMonitor()

    # The learning module can also define its own specific callbacks
    callbacks = pl_module.callbacks + [
        checkpoint_callback,
        lr_monitor
    ]

    # Using from_argparse_args enables to use any standard parameter of the
    # lightning Trainer class without having to manually add them to the parser.
    trainer = Trainer.from_argparse_args(
        args,
        logger=tensorboard,
        profiler='simple',
        callbacks=callbacks,
        benchmark=True,
        max_epochs=100
    )

    trainer.fit(model=pl_module, datamodule=pl_datamodule)


if __name__ == "__main__":

    main()