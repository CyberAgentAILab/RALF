import argparse
import logging
import torch

from pathlib import Path
from PIL import Image
from typing import Any
import os
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import normalize
import numpy as np


from models.saliency.basnet import BASNet, RescaleT, ToTensorLab
from models.saliency.isnet import ISNetDIS

logger = logging.getLogger(__name__)


WEIGHT_ROOT = Path(__file__).resolve().parent / "pretrained_weights" / "saliency_detection"  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Input file path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output prefix.",
    )
    parser.add_argument(
        "--input_ext",
        type=str,
        help="Limit the type of input file extension.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["isnet", "basnet"],
        default="isnet",
    )

    args = parser.parse_args()
    logger.info(f"{args=}")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.algorithm == "isnet":
        tester = ISNetSaliencyTester()
    elif args.algorithm == "basnet":
        tester = BASNetSaliencyTester()
    else:
        raise NotImplementedError

    pattern = f"*.{args.input_ext}" if args.input_ext else "*"
    for input_path in Path(args.input_dir).glob(pattern):
        if not input_path.is_file():
            continue

        image = Image.open(input_path).convert("RGB")
        width, height = image.size

        pred = tester(image)
        pred = torch.squeeze(F.interpolate(pred, (height, width), mode="bilinear"), 0)
        pred = _norm_pred(pred)

        output_path = output_dir / input_path.name
        logger.info(f"{input_path=} {output_path=}")
        with output_path.open("wb") as f:
            save_image(pred, f)


class _SaliencyTester:  # type: ignore
    def __init__(self) -> None:
        self._model: nn.Module = nn.Identity()  # to be overwritten
        self._ckpt_path: str = ""  # to be overwritten

    def setup_model(self, model: nn.Module) -> None:
        model.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"))
        model.eval()
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        self._model = model

    def __call__(self, image: Image) -> None:
        raise NotImplementedError


class ISNetSaliencyTester(_SaliencyTester):  # type: ignore
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self._transform = ToTensor()
        self._input_size = (1024, 1024)
        self._ckpt_path = str(WEIGHT_ROOT / "isnet-general-use.pth")
        self.setup_model(ISNetDIS())  # type: ignore

    @torch.no_grad()
    def __call__(self, image: Image) -> Tensor:
        # preprocess
        width, height = image.size
        img = self._transform(image).unsqueeze(0)
        img = F.interpolate(img, self._input_size, mode="bilinear")
        img = normalize(img, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))

        # prediction
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda"))
        pred = self._model(img)[0][0].cpu().detach()
        assert list(pred.size()) == [1, 1, 1024, 1024]

        return pred
        # self.postprocess_and_save_image(pred=pred, path=path, width=width, height=height)


class BASNetSaliencyTester(_SaliencyTester):  # type: ignore
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # note: this transforms takes and returns numpy in np.uint8 with size (H, W, C)
        self._transform = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])  # type: ignore
        self._ckpt_path = str(WEIGHT_ROOT / "gdi-basnet.pth")
        self.setup_model(BASNet(3, 1))  # type: ignore

    @torch.no_grad()
    def __call__(self, img_pil: Image) -> Tensor:
        # preprocess
        img_npy = np.array(img_pil, dtype=np.uint8)
        width, height = img_pil.size
        assert img_npy.shape[-1] == 3
        label_npy = np.zeros((height, width), dtype=np.uint8)
        img = self._transform({"image": img_npy, "label": label_npy})["image"]
        img = img.float().unsqueeze(0)

        # prediction
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda"))
        pred = self._model(img)[0].cpu().detach()[:, 0].unsqueeze(0)
        assert list(pred.size()) == [1, 1, 256, 256]

        return pred
        # self.postprocess_and_save_image(pred=pred, path=path, width=width, height=height)


def _norm_pred(d: Tensor) -> Tensor:
    ma = torch.max(d)
    mi = torch.min(d)
    # division while avoiding zero division
    dn = (d - mi) / ((ma - mi) + torch.finfo(torch.float32).eps)
    return dn


if __name__ == "__main__":
    main()
