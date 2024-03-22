import random
from typing import Any, Optional, Union

import fsspec
import numpy as np
import torch
from torch import Tensor


def is_run_on_local(dir_name: str) -> bool:
    """
    Check if script is run local machine or not by the given directory / file name.
    """
    fs, _ = fsspec.core.url_to_fs(dir_name)
    return isinstance(fs, fsspec.implementations.local.LocalFileSystem)


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x: list[Union[int, float]]) -> list[int]:
    assert isinstance(x, list) and isinstance(x[0], (int, float))
    return sorted(range(len(x)), key=x.__getitem__)


def is_dict_of_list(x: Any) -> bool:
    if isinstance(x, dict):
        return all(isinstance(v, list) for v in x.values())
    else:
        return False


def dict_of_list_to_list_of_dict(dl: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def is_list_of_dict(x: Any) -> bool:
    if isinstance(x, list):
        return all(isinstance(d, dict) for d in x)
    else:
        return False


def list_of_dict_to_dict_of_list(ld: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def pad(data: list[Any], max_seq_length: int) -> list[Any]:
    assert len(data) > 0
    value = data[0]
    if isinstance(value, bool):
        pad_value = False
    elif isinstance(value, int):
        pad_value = 0
    elif isinstance(value, float):
        pad_value = 0.0
    else:
        raise NotImplementedError

    n = len(data)
    assert n <= max_seq_length
    return data + [pad_value] * (max_seq_length - n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def convert_xywh_to_ltrb(
    bbox: Union[Tensor, np.ndarray, list[float]]
) -> Union[list[Tensor], list[np.ndarray], list[float]]:
    assert len(bbox) == 4
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def batch_shuffle_index(
    batch_size: int,
    feature_length: int,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.LongTensor:
    """
    Note: masked part may be shuffled because of
    unpredictable behaviour of sorting [inf, ..., inf]
    """
    if mask:
        assert list(mask.size()) == [batch_size, feature_length]
    scores = torch.rand((batch_size, feature_length))
    if mask:
        scores[~mask] = float("Inf")
    indices: torch.LongTensor = torch.sort(scores, dim=1)[1]
    return indices
