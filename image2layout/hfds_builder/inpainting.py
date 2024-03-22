import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from helpers.cgl import read_cgl
from helpers.pku import read_pku
from helpers.util import Sample
from models.inpainting import SimpleLama
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="pku",
        choices=["pku", "cgl"],
        help="Kind of pipeline to invoke.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        type=str,
        help="Input file path.",
    )
    parser.add_argument(
        "--log_level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    output_dir = Path(args.dataset_root) / "image" / "train" / "input"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "pku":
        reader = read_pku
    elif args.dataset_type == "cgl":
        reader = read_cgl
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    lama = SimpleLama()

    stats = defaultdict(int)  # type: ignore
    for sample in tqdm(reader(args.dataset_root), desc="Pre-processing dataset"):
        split, file_name = sample.identifier.split("/")
        file_name = file_name.replace(
            ".jpg", ".png"
        )  # even if input is JPG, output is PNG
        output_path = str(output_dir / file_name)
        if split == "test":
            # no need to inpaint since test set is a layout-free image
            stats["already_clean"] += 1
            continue

        if os.path.exists(output_path):
            logger.debug(f"Skipping {output_path=} since it already exists.")
            stats["already_exists"] += 1
            continue

        image_path = os.path.join(
            args.dataset_root, "image", "train", "original", file_name
        )
        image = Image.open(image_path).convert("RGB")
        mask = _get_mask(sample)
        result = lama(image, mask)
        result.resize(image.size).save(output_path)
        stats["processed"] += 1

    logger.info(f"{stats=}")


def _dilate_mask(
    original_mask: np.ndarray, kernel_size: int = 5, iterations: int = 8
) -> np.ndarray:
    assert kernel_size > 0 and kernel_size % 2 == 1
    assert iterations > 0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(
        original_mask.astype("uint8"), kernel, iterations=iterations
    )
    return dilated_mask.astype("float32")  # typing: ignore


def _get_mask(sample: Sample) -> Image:
    """
    Generate a mask image for inpainting from the annotation.
    The image should be
    - in (H, W) format
    - normalized in [0.0, 1.0] range (np.float32)
    """
    H, W = sample.image_height, sample.image_width
    canvas = np.zeros((H, W, 1), dtype=np.float32)

    # draw bounding boxes
    for element in sample.elements:
        coord = element.coordinates
        x1 = math.floor(W * coord.left)
        x2 = math.ceil(W * coord.right)
        y1 = math.floor(H * coord.top)
        y2 = math.ceil(H * coord.bottom)

        cv2.rectangle(
            canvas,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=(1, 1, 1),
            thickness=-1,
            lineType=cv2.LINE_4,
            shift=0,
        )

    canvas = _dilate_mask(canvas)
    return Image.fromarray((canvas * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
