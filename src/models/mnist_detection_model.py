from math import nan
from typing import Any, List

import numpy as np
import torch
from pl_bolts.losses.object_detection import iou_loss
from pl_bolts.metrics.object_detection import iou
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.detection.map import MAP
from torchvision import models
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


class MnistDetectionLitModel(LightningModule):
    """
    LightningModule for mnist object detection.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_classes: int = 80,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        batch_size: int = 4,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor):
        pass

    def training_step(self, batch: Any, batch_idx: int):

        imgs, tars, _ = batch

        loss = 0

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):

        imgs, tars, _ = batch
        
        preds = []
        targets = []

        return {"preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        imgs, tars, fnames = batch
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple."""
        pass