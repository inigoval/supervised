import wandb
import pytorch_lightning as pl
import logging
import torch

from pytorch_lightning.callbacks import LearningRateMonitor

from paths import create_path, Path_Handler
from classifier import Supervised
from dataloading import MiraBest_DataModule
from config import load_config

# TODO put elsewhere
# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Profile_PyTorch_Code.ipynb#scrollTo=qRoUXZdtJIUD


def run_supervised_training(config, datamodule, wandb_logger):

    pl.seed_everything(config["seed"])

    ## Creates experiment path if it doesn't exist already ##
    experiment_dir = config["files"] / config["run_id"]
    create_path(experiment_dir)

    ## Initialise checkpoint ##
    checkpoint = pl.callbacks.ModelCheckpoint(
        # **checkpoint_mode[config["evaluation"]["checkpoint_mode"]],
        # mode = 'max',
        # monitor="val/acc",
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=experiment_dir / "checkpoints",
        save_last=True,
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        # filename="{epoch}-{step}-{loss_to_monitor:.4f}",  # filename may not work here TODO
        filename="model",
        save_weights_only=True,
    )

    ## Initialise callbacks ##
    callbacks = [checkpoint]

    # add learning rate monitor, only supported with a logger
    if wandb_logger is not None:
        # change to step, may be slow
        callbacks += [LearningRateMonitor(logging_interval="epoch")]

    logging.info(f"Threads: {torch.get_num_threads()}")

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        **config["trainer"],
        max_epochs=config["model"]["n_epochs"],
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=200,
        # max_steps = 200  # TODO temp
    )

    # Initialise model #
    model = Supervised(config)

    # profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
    # profile_art.add_file(glob.glob(str(experiment_dir / "*.pt.trace.json"))[0], "trace.pt.trace.json")
    # wandb.run.log_artifact(profile_art)

    # Train model #
    trainer.fit(model, datamodule)
    trainer.test(model, dataloaders=datamodule)

    return checkpoint, model


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    ## Load up config from yml files ##
    config = load_config()

    wandb.init(project=config["project_name"])
    config["run_id"] = str(wandb.run.id)

    path_dict = Path_Handler()._dict()

    wandb_logger = pl.loggers.WandbLogger(
        project=config["project_name"],
        # and will then add e.g. run-20220513_122412-l5ikqywp automatically
        save_dir=path_dict["files"] / config["run_id"],
        # log_model="True",
        # reinit=True,
        config=config,
    )

    config["files"] = path_dict["files"]

    ## Run pretraining ##
    for seed in range(config["finetune"]["iterations"]):
        config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=f"{config['project_name']}_finetune", config=config)

        logger = pl.loggers.WandbLogger(
            project=config["project_name"],
            save_dir=path_dict["files"] / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        ## Initialise data and run set up ##
        # data = datasets[config["dataset"]](config)
        datamodule = MiraBest_DataModule(config)

        checkpoint, model = run_supervised_training(config, datamodule, logger)
        wandb.save(checkpoint.best_model_path)
        logger.experiment.finish()

    # wadnb.save()

    wandb_logger.experiment.finish()


if __name__ == "__main__":

    main()
