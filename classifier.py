import torch
import torch.nn as nn
import lightly
import copy
import torchmetrics as tm
import pytorch_lightning as pl

from math import cos, pi
from utilities import _optimizer
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import torch.nn as nn

from architectures.resnet import _get_resnet

## Needs to be rewritten ##


class Supervised(pl.LightningModule):
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        self.encoder = _get_resnet(self.config["model"]["architecture"])
        self.head = nn.Linear(config["model"]["features"], config["data"]["classes"])

        self.val_acc = tm.Accuracy(average="micro", threshold=0)

        self.test_acc = tm.Accuracy(average="micro", threshold=0)
        self.test_f1 = tm.F1Score(num_classes=2, average="none")
        self.test_precision = tm.Precision(num_classes=2, average="none")
        self.test_recall = tm.Recall(num_classes=2, average="none")

    def forward(self, x):
        # dimension (batch, features), features from config e.g. 512
        return self.encoder(x)

    def predict(self, x):
        x = self.encoder(x).squeeze()
        x = self.head(x)
        x = F.softmax(x, dim=-1)
        return x

    def on_fit_start(self):
        self.config["data"]["mu"] = self.trainer.datamodule.mu
        self.config["data"]["sig"] = self.trainer.datamodule.sig

        # Log size of data-sets #
        logging_params = {f"n_{key}": len(value) for key, value in self.trainer.datamodule.data.items()}
        self.logger.log_hyperparams(logging_params)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.n_layers else 0)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.val_acc(preds, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.log("test_f1", self.test_f1)
        self.log("test_precision", self.test_precision)
        self.log("test_recall", self.test_recall)

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.head.parameters())

        return _optimizer(params, self.config)
