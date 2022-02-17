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
        num_classes: int = 11,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        trainable_backbone_layers: int = 0,
        batch_size: int = 4,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.hparams.num_classes
        )

        self.val_map = MAP()
        self.test_map = MAP(class_metrics=True)

        # for logging best so far validation map
        self.val_map_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):

        images, targets, _ = batch

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):

        images, targets, _ = batch

        # fastercnn module just returns the predicitions no loss so we calc map 
        preds = self.model(images)

        val_map = self.val_map(preds, targets)
        self.log("val/map", val_map["map"], on_step=False, on_epoch=True, prog_bar=False)

        return {"preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):

        preds = []
        targets = []
        for pt_dict in outputs:
            for pred, target in zip(pt_dict["preds"], pt_dict["targets"]):
                preds.append(pred)
                targets.append(target)

        val_map_dict = self.val_map(preds, targets)

        self.log("val/map_dict", val_map_dict, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        images, targets, file_names = batch

        preds = self.model(images)

        return {
            "test_images": images,
            "test_gt": targets,
            "test_outs": preds,
            "file_names": file_names,
        }

    def test_epoch_end(self, outputs: List[Any]):
        preds = []
        targets = []
        for pt_dict in outputs:
            for pred, target in zip(pt_dict["test_outs"], pt_dict["test_gt"]):
                preds.append(pred)
                targets.append(target)

        test_map_dict = self.test_map(preds, targets)

        self.log("test/map_dict", test_map_dict)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.val_map.reset()
        self.test_map.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.SGD(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
        )
