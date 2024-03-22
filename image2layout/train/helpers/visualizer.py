import logging
import pathlib
from functools import lru_cache

# from graphviz import Graph
from typing import Optional, Union

import datasets as ds
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from einops import rearrange
from PIL import Image, ImageDraw
from torch import BoolTensor, LongTensor, Tensor

from .util import convert_xywh_to_ltrb

logger = logging.getLogger(__name__)
TENSOR_TO_PIL = T.ToPILImage()


@lru_cache(maxsize=32)
def get_colors(n_colors: int) -> list[tuple[int, int, int]]:
    colors = sns.color_palette("husl", n_colors=n_colors)
    colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
    return colors  # type: ignore


def render(
    prediction: dict[str, Tensor],
    label_feature: ds.ClassLabel,
    bg_key: str = "image",
    use_grid: bool = True,
) -> Optional[Tensor]:
    colors = get_colors(label_feature.num_classes)
    batch_images = prediction[bg_key]
    if bg_key == "saliency":
        # (1, H, W) -> (3, H, W) or (B, 1, H, W) -> (B, 3, H, W)
        batch_images = torch.cat([batch_images for _ in range(3)], dim=-3)

    keys = ["center_x", "center_y", "width", "height"]
    batch_bboxes = torch.stack([prediction[key] for key in keys], dim=-1)
    image = save_image(
        batch_images=batch_images,
        batch_bboxes=batch_bboxes,
        batch_labels=prediction["label"],
        batch_masks=prediction["mask"],
        colors=colors,
        use_grid=use_grid,
    )
    if image is not None:
        return image
    else:
        return None


def convert_layout_to_image(
    background: Tensor,
    bboxes: Tensor,
    labels: LongTensor,
    colors: list[tuple[int, int, int]],
    canvas_size: Optional[tuple[int, int]] = (240, 160),
) -> Image:
    C, H, W = background.size()
    assert background.size(0) == C
    img = TENSOR_TO_PIL(background[:3, ...])
    draw = ImageDraw.Draw(img, "RGBA")

    # draw from larger boxes
    a = [b[2] * b[3] for b in bboxes]
    indices = sorted(range(len(a)), key=lambda i: a[i], reverse=True)

    for i in indices:
        bbox, label = bboxes[i], labels[i]
        if isinstance(label, LongTensor):
            label = label.item()

        c_fill = colors[label] + (160,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)

        draw.rectangle([x1, y1, x2, y2], outline=colors[label], fill=c_fill)

    return img


def save_image(
    batch_images: Tensor,
    batch_bboxes: Tensor,
    batch_labels: LongTensor,
    batch_masks: BoolTensor,
    colors: list[tuple[int, int, int]],
    out_path: Optional[Union[pathlib.PosixPath, str]] = None,
    canvas_size: tuple[int, int] = (60, 40),
    nrow: Optional[int] = None,
    batch_resources: Optional[dict] = None,
    use_grid: bool = False,
    **kwargs,
) -> Optional[Tensor]:
    if batch_labels.dim() == 1:
        # if the data is not batched, add batch dimension
        batch_images = rearrange(batch_images, "c h w -> 1 c h w")
        batch_labels = rearrange(batch_labels, "n -> 1 n")
        batch_bboxes = rearrange(batch_bboxes, "n x -> 1 n x")
        batch_masks = rearrange(batch_masks, "n -> 1 n")
    assert batch_images.dim() == 4 and batch_masks.dim() == 2
    assert batch_bboxes.dim() == 3 and batch_labels.dim() == 2

    if isinstance(out_path, pathlib.PosixPath):
        out_path = str(out_path)

    imgs = []
    B = batch_bboxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        background = batch_images[i]
        mask_i = batch_masks[i]
        bboxes = batch_bboxes[i][mask_i]
        labels = batch_labels[i][mask_i]
        if batch_resources:
            resources = {k: v[i] for (k, v) in batch_resources.items()}
            img = convert_layout_to_image(
                background, bboxes, labels, colors, canvas_size, resources
            )
        else:
            img = convert_layout_to_image(
                background, bboxes, labels, colors, canvas_size
            )
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    if out_path:
        vutils.save_image(image, out_path, normalize=False, nrow=nrow)
    else:
        if use_grid:
            return vutils.make_grid(image, normalize=False, nrow=nrow)
        else:
            return image

def mask_out_bbox_area(images, bboxes):
    # images: [batch_size, 3, height, width]
    # bboxes: [batch_size, max_elem, 4], with [top-left x, top-left y, bottom-right x, bottom-right y]

    batch_size, _, height, width = images.shape
    mask = torch.ones(batch_size, height, width).to(images.device)
    max_elem = bboxes.size(1)

    y_indices, x_indices = torch.meshgrid(
        torch.arange(0, height), torch.arange(0, width)
    )
    y_indices = y_indices.unsqueeze(0).repeat(batch_size, 1, 1).to(images.device)
    x_indices = x_indices.unsqueeze(0).repeat(batch_size, 1, 1).to(images.device)

    for j in range(max_elem):
        # De-normalize the bounding box coordinates
        x1 = (bboxes[:, j, 0] * width).long().unsqueeze(1).unsqueeze(2)
        y1 = (bboxes[:, j, 1] * height).long().unsqueeze(1).unsqueeze(2)
        x2 = (bboxes[:, j, 2] * width).long().unsqueeze(1).unsqueeze(2)
        y2 = (bboxes[:, j, 3] * height).long().unsqueeze(1).unsqueeze(2)

        # Check if the bounding box is not padding (all zeros)
        valid_bbox = (x1 + y1 + x2 + y2) != 0

        current_mask = (
            (x_indices < x1) + (x_indices >= x2) + (y_indices < y1) + (y_indices >= y2)
        ) > 0
        mask = mask * current_mask
        mask[valid_bbox.squeeze()] = 1 - (1 - mask[valid_bbox.squeeze()]).clamp(0, 1)

    return images * mask.unsqueeze(1)
