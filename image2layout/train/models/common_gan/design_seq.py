from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1[..., :4])
    area2 = box_area(boxes2[..., :4])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:4], boxes2[:, 2:4])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return (iou, union)


def reorder(cls: list[float], box: Tensor, o="xyxy", max_elem=None) -> list[int]:
    """
    Args:
        cls (list[float]): hoge
        box (torch.Tensor): hoge
    Return:

    """

    if isinstance(box, np.ndarray):
        box = torch.tensor(box).float()

    if o == "cxcywh":
        box = box_cxcywh_to_xyxy(box)
    if max_elem is None:
        max_elem = len(cls)

    order = []

    # convert
    cls_np = np.array(cls)
    area = box_area(box)
    if isinstance(area, Tensor):
        area = area.detach().cpu().numpy()
    # order_area = sorted(list(enumerate(area)), key=lambda x: x[1], reverse=True)
    iou, _ = box_iou(box, box)  # (10, 10)

    # arrange
    text = np.where(cls_np == 1)[0]
    logo = np.where(cls_np == 2)[0]
    deco = np.where(cls_np == 3)[0]

    # print(f"{text=}, {logo=}, {deco=}")

    order_text = sorted(
        np.array(list(enumerate(area)))[text].tolist(), key=lambda x: x[1], reverse=True
    )
    order_deco = sorted(
        np.array(list(enumerate(area)))[deco].tolist(), key=lambda x: x[1]
    )

    # deco connection
    connection = {}
    reverse_connection = {}
    for idx, _ in order_deco:
        idx = int(idx)
        con = []
        for idx_ in logo:
            idx_ = int(idx_)
            if iou[idx, idx_]:
                connection[idx_] = idx
                con.append(idx_)
        for idx_ in text:
            idx_ = int(idx_)
            if iou[idx, idx_]:
                connection[idx_] = idx
                con.append(idx_)
        for idx_ in deco:
            idx_ = int(idx_)
            if idx == idx_:
                continue
            if iou[idx, idx_]:
                if idx_ not in connection:
                    connection[idx_] = [idx]
                else:
                    connection[idx_].append(idx)
                con.append(idx_)
        reverse_connection[idx] = con

    # print(f"{reverse_connection}")

    # reorder
    for idx in logo:
        if idx in connection:
            d = connection[idx]
            d_group = reverse_connection[d]
            for idx_ in d_group:
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            order.append(idx)
    # print(f"A: {order=}")
    for idx, _ in order_text:
        if len(order) >= max_elem:
            break
        if idx in connection:
            d = connection[idx]
            d_group = reverse_connection[d]
            for idx_ in d_group:
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            order.append(idx)

    unsed_deco = list(set(deco.tolist()) - set(order))
    order += unsed_deco

    if len(order) < max_elem:
        non_obj = np.where(cls_np == 0)[0]
        order.extend(non_obj)

    order = list(map(int, order))
    return order[: min(len(cls), max_elem)]
