import sys
import torchvision.transforms as T
import pytorch_lightning as pl

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import DataLoader

from paths import Path_Handler
from astroaugmentations.datasets.MiraBest_F import MBFRFull, MBFRConfident


class Supervised_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        # override default paths via config if desired
        paths = Path_Handler(**config.get("paths_to_override", {}))
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.config = config

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"], self.config["batch_size"], shuffle=True, **self.config["dataloader"]
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data["val"], self.config["val_batch_size"], shuffle=False, **self.config["dataloader"]
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.data["test"], self.config["val_batch_size"], shuffle=False, **self.config["dataloader"]
        )
        return loader


# TODO get aug parameters like center_crop from loaded model
class MiraBest_DataModule(Supervised_DataModule):
    def __init__(self, config):
        super().__init__(config)
        self.mu, self.sig = (0.008008896,), (0.05303395,)

        self.T_train = T.Compose(
            [
                T.RandomRotation(180),
                T.CenterCrop(config["center_crop_size"]),
                T.RandomResizedCrop(config["center_crop_size"], scale=(0.8, 1)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize((0.008008896,), (0.05303395,)),
            ]
        )

        self.T_test = T.Compose(
            [
                T.CenterCrop(["center_crop_size"]),
                T.ToTensor(),
                T.Normalize((0.008008896,), (0.05303395,)),
            ]
        )

    def prepare_data(self):
        MBFRFull(self.path, train=False, download=True)
        MBFRFull(self.path, train=True, download=True)

    def setup(self, stage=None):
        data_dict = {
            "root": self.path,
            "aug_type": "torchvision",
            "test_size": self.config["data"]["test_size"],
            "seed": self.config["seed"],
        }

        self.data["train"] = MBFRConfident(**data_dict, train=True, transform=self.T_train)
        self.data["val"] = MBFRConfident(**data_dict, train=True, transform=self.T_test)
        self.data["test"] = MBFRConfident(**data_dict, train=False, transform=self.T_test)
