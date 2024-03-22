import random
from enum import IntEnum
from itertools import combinations

import torch
from image2layout.train.global_variables import GEO_KEYS

from .util import convert_xywh_to_ltrb


class RelSize(IntEnum):
    UNKNOWN = 0
    SMALLER = 1
    EQUAL = 2
    LARGER = 3


class RelLoc(IntEnum):
    UNKNOWN = 4
    LEFT = 5
    TOP = 6
    RIGHT = 7
    BOTTOM = 8
    CENTER = 9


RELATIVE_RELATION = {
    RelLoc.LEFT: RelLoc.RIGHT,
    RelLoc.RIGHT: RelLoc.LEFT,
    RelLoc.TOP: RelLoc.BOTTOM,
    RelLoc.BOTTOM: RelLoc.TOP,
    RelLoc.CENTER: RelLoc.CENTER,
    RelLoc.UNKNOWN: RelLoc.UNKNOWN,
    RelSize.SMALLER: RelSize.LARGER,
    RelSize.LARGER: RelSize.SMALLER,
    RelSize.EQUAL: RelSize.EQUAL,
    RelSize.UNKNOWN: RelSize.UNKNOWN,
}


class RelElement(IntEnum):
    A = 10
    B = 11
    C = 12
    D = 13
    E = 14
    F = 15
    G = 16
    H = 17
    I = 18
    J = 19
    K = 20


REL_SIZE_ALPHA = 0.1


def detect_size_relation(b1: list[float], b2: list[float]) -> RelSize:
    """
    Args:
        b1: xywh of bbox1
        b2: xywh of bbox2
    Return:
        RelSize
    """
    a1 = b1[2] * b1[3]
    a2 = b2[2] * b2[3]
    alpha = REL_SIZE_ALPHA
    if (1 - alpha) * a1 < a2 < (1 + alpha) * a1:
        return RelSize.EQUAL
    elif a1 < a2:  # b2 > b1
        return RelSize.LARGER
    else:  # B1 > B2
        return RelSize.SMALLER


def detect_loc_relation_between_elements(
    bbox1: list[float], bbox2: list[float]
) -> RelLoc:
    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox1)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox2)

    if b2 <= t1:
        # bbox２ is on the top of bbox1
        return RelLoc.TOP
    elif b1 <= t2:
        # bbox２ is on the bottom of bbox1
        return RelLoc.BOTTOM
    elif r2 <= l1:
        # bbox２ is on the left of bbox1
        return RelLoc.LEFT
    elif r1 <= l2:
        # bbox２ is on the right of bbox1
        return RelLoc.RIGHT
    else:
        # Overlap two bounding boxes
        return RelLoc.CENTER


def detect_loc_relation_between_element_and_canvas(bbox: list[float]) -> RelLoc:
    yc = bbox[1]  # center_y
    if yc < 1.0 / 3:
        return RelLoc.TOP
    elif yc < 2.0 / 3:
        return RelLoc.CENTER
    else:
        return RelLoc.BOTTOM


def compute_relation(
    batch: dict[str, torch.Tensor], edge_ratio: float = 0.1
) -> dict[str, torch.Tensor]:
    B, S = batch["label"].size()
    tmp_batch = {}
    tmp_batch["label"] = torch.cat(
        [torch.full((B, 1), fill_value=-1), batch["label"]], dim=1
    )
    tmp_batch["mask"] = torch.cat(
        [torch.full((B, 1), fill_value=True), batch["mask"]], dim=1
    )
    for key in ["center_x", "center_y"]:
        tensor = torch.full((B, 1), fill_value=0.5)
        tmp_batch[key] = torch.cat([tensor, batch[key]], dim=1)
    for key in ["width", "height"]:
        tensor = torch.full((B, 1), fill_value=1.0)
        tmp_batch[key] = torch.cat([tensor, batch[key]], dim=1)

    rel_unk = (1 << RelSize.UNKNOWN) | (1 << RelLoc.UNKNOWN)
    E = (S + 1) * (S + 2) // 2

    edge_indexs = torch.full((B, E, 2), fill_value=-1, dtype=torch.long)
    edge_attributes = torch.full((B, E), fill_value=rel_unk, dtype=torch.long)

    num_element = tmp_batch["mask"].sum(dim=1)
    for b in range(B):
        n = num_element[b]
        cnt = 0
        for i, j in combinations(range(S + 1), 2):
            # ignore relation between dummy element and any
            if n <= i or n <= j:
                continue

            # stochasticly drop relations
            if random.random() > edge_ratio:
                continue

            is_canvas = i == 0
            bi = [tmp_batch[key][b][i] for key in GEO_KEYS]
            bj = [tmp_batch[key][b][j] for key in GEO_KEYS]

            rel_size = 1 << detect_size_relation(bi, bj)
            if is_canvas:
                rel_loc = 1 << detect_loc_relation_between_element_and_canvas(bj)
            else:
                rel_loc = 1 << detect_loc_relation_between_elements(bi, bj)
            rel = rel_size | rel_loc

            edge_indexs[b, cnt, 0] = i
            edge_indexs[b, cnt, 1] = j
            edge_attributes[b, cnt] = rel
            cnt += 1

    return {
        "edge_indexes": edge_indexs,
        "edge_attributes": edge_attributes,
    }
