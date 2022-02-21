import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

from src.models.mnist_detection_model import MnistDetectionLitModel

class_dict = {
    10: "zero",
    1:  "one",
    2:  "two",
    3:  "three",
    4:  "four",
    5:  "five",
    6:  "six",
    7:  "seven",
    8:  "eight",
    9:   "nine",
}


def _visualize_bbox(img, bbox, class_name, score, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image"""
    if score > 0.99:
        x_min, y_min, x_max, y_max = map(int,bbox)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(
            class_dict[class_name], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )
        cv2.rectangle(
            img,
            (x_min, y_min - int(1.3 * text_height)),
            (x_min + text_width, y_min),
            (255, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            text=class_dict[class_name],
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )
    return img


def _visualize(image, bboxes, category_names, scores):
    img = image.copy()
    for bbox, class_name, score in zip(bboxes, category_names, scores):
        img = _visualize_bbox(img, bbox, class_name, score)
    return img


def predict():
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    CKPT_PATH = "./best.ckpt"

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = MnistDetectionLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("./data/mnist_detection_data/test/16.png").convert("RGB")

    # preprocess
    transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
    image_np = np.array(img)
    transformed = transforms(image=image_np)
    img = transformed["image"]  # reshape to form batch of size 1

    img = torch.unsqueeze(img, dim=0)

    # inference
    output = trained_model(img)
    print(output)

    img = _visualize(
        image_np,
        output[0]["boxes"].numpy(),
        output[0]["labels"].numpy(),
        output[0]["scores"].numpy(),
    )

    print("done")


if __name__ == "__main__":
    predict()
