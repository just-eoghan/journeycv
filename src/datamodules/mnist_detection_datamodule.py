import datetime
import os
from typing import Optional, Tuple

import albumentations as A
import boto3
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm

from src.datamodules.datasets.thermal_dataset import ThermalDataset

s3 = boto3.resource("s3")
cw = boto3.client("cloudwatch")


class Collater:
    # https://shoarora.github.io/2020/02/01/collate_fn.html
    def __call__(self, batch):
        return tuple(zip(*batch))


class MnistDetectionDataModule(LightningDataModule):
    """
    MnistDetectionDataModule for Mnist object detection.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/mnist_detection_data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 10
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = None
        self.notransforms = None
        self.collater = Collater()

        # self.dims is returned when you call datamodule.size()
        self.dims = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return 

    def val_dataloader(self):
        return 

    def test_dataloader(self):
        return 
