# Credit: https://github.com/david-wb/softargmax/blob/master/softargmax.py

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def softargmax2d(input: Tensor, beta: int = 100) -> Tensor:
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w), np.linspace(0, 1, h), indexing="xy"
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).type_as(input)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).type_as(input)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result


def softargmax1d(input: Tensor, beta: int = 100) -> Tensor:
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).type_as(input)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def differentiable_round(input: Tensor) -> Tensor:
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out
