import datetime
import os
from typing import Optional, Tuple

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
from urllib import request
import pathlib
import pickle
import gzip
import numpy as np

from src.datamodules.datasets.mnist_detection_dataset import MnistDetectionDataset

import src.datamodules.mnist_generate.mnist as mnist
import src.datamodules.mnist_generate.generate_data as generate_data

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
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 11,
        num_generated_train: int = 10000,
        num_generated_test: int = 1000,
        generated_min_digit_size: int = 15,
        generated_max_digit_size: int = 100,
        generated_image_size: int = 300,
        max_digits_per_generated_image: int = 20,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = A.Compose(
            [
                A.Normalize(),
                A.HorizontalFlip(p=0.5),
                A.Blur(blur_limit=3, p=0.15),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        )

        self.notransforms = A.Compose(
            [
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        )

        self.collater = Collater()

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 300, 300)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        if not os.path.exists(self.hparams.data_dir):
            print("downloading dataset")
            X_train, Y_train, X_test, Y_test = mnist.load()
            for dataset, (X, Y) in zip(["train", "test"], [[X_train, Y_train], [X_test, Y_test]]):
                num_images = self.hparams.num_generated_train if dataset == "train" else self.hparams.num_generated_test
                generate_data.generate_dataset(
                    dataset,
                    pathlib.Path(self.hparams.data_dir, dataset),
                    num_images,
                    self.hparams.generated_max_digit_size,
                    self.hparams.generated_min_digit_size,
                    self.hparams.generated_image_size,
                    self.hparams.max_digits_per_generated_image,
                    X,
                    Y)             
            return 


    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = MnistDetectionDataset(
                self.hparams.data_dir + "train",
                self.hparams.data_dir + "train.json",
                transform=self.transforms,
            )

            self.data_test = MnistDetectionDataset(
                self.hparams.data_dir + "test",
                self.hparams.data_dir + "test.json",
                transform=self.notransforms,
            )

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=(int(self.hparams.num_generated_train * 0.7), int(self.hparams.num_generated_train * 0.3)),
                generator=torch.Generator().manual_seed(42),
            )

            self.data_val.dataset.transform = self.notransforms

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # https://github.com/pytorch/vision/issues/2624#issuecomment-681811444
            collate_fn=self.collater,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collater,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collater,
        )
