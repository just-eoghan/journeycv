import os
import os.path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.ops import box_convert


class MnistDetectionDataset(VisionDataset):
    """
     Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)


    def _load_image(self, id: int) -> Tuple[Image.Image, str]:
        pass

    def _load_target(self, id: int) -> List[Any]:
        pass

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        pass

    def __len__(self) -> int:
        return len(self.ids)
