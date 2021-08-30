import configparser
import pathlib as path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from idao.data_module import IDAODataModule
from idao.model import SimpleConv

#import mlflow stuff
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def get_free_gpu():
    """
    Returns the index of the GPU with the most free memory.
    Different from lightning's auto_select_gpus, as it provides the most free GPU, not an absolutely free.
    """
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
    nvmlInit()

    return np.argmax([
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
        for i in range(nvmlDeviceGetCount())
    ])


def print_auto_logged_info(r):
    print("run_id: {}".format(r.info.run_id))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))


def trainer(logger, mode: ["classification", "regression"], cfg, dataset_dm, learning_rate):
    model = SimpleConv(learning_rate=learning_rate, mode=mode)
    if cfg.getboolean("TRAINING", "UseGPU"):
        gpus = [get_free_gpu()]
    else:
        gpus = None
    if mode == "classification":
        epochs = cfg["TRAINING"]["ClassificationEpochs"]
    else:
        epochs = cfg["TRAINING"]["RegressionEpochs"]

    # Build dataloaders
    train = dataset_dm.train_dataloader()
    valid = dataset_dm.val_dataloader()

    # Build callbacks
    check_dir = "./checkpoints/" + mode + "/" + logger.name + "/version_" + str(logger.version)
    print("Will save checkpoints at ", check_dir)

    checkpoint_callback = ModelCheckpoint(dirpath=check_dir,
                                              filename='{epoch}-{valid_loss:.2f}',
                                              monitor= 'valid_acc_epoch',
                                              mode='max',
                                              save_top_k=1,
                                              )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=gpus,
        max_epochs=int(epochs),
        progress_bar_refresh_rate=1,
        weights_save_path=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]).joinpath(
            mode
        ),
        default_root_dir=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]),
        logger=logger
    )

    #mlflow.set_tracking_uri("databricks")
    # Note: on Databricks, the experiment name passed to set_experiment must be a valid path
    # in the workspace, like '/Users/<your-username>/my-experiment'. See
    # https://docs.databricks.com/user-guide/workspace.html for more info.
    #mlflow.set_experiment("/my-experiment")


    #mlflow.set_experiment("model5")
    mlflow.pytorch.autolog()

    # Train the model ⚡
    with mlflow.start_run() as run:
        trainer.fit(model, train, valid)
        #mlflow.log_param("epochs", epochs)
        #mlflow.pytorch.log_model(model,"model5")
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


def main():
    seed_everything(666)
    config = configparser.ConfigParser()
    config.read("./config.ini")

    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=int(config["TRAINING"]["BatchSize"]), cfg=config
    )
    dataset_dm.prepare_data()
    print(dataset_dm.dataset)
    dataset_dm.setup()


    for mode in ["classification"]:
        print(f"Training for {mode}")
        logger = TensorBoardLogger('runs', 'classification-CNN', log_graph=True)
        trainer(logger, mode, cfg=config, dataset_dm=dataset_dm, learning_rate=1e-3)


if __name__ == "__main__":
    main()
