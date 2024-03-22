import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-30
LOG_EPS = math.log(1e-30)


def mean_except_batch(x: Tensor, num_dims: int = 1) -> Tensor:
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)


def log_1_min_a(a: Tensor) -> Tensor:
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a: Tensor, b: Tensor) -> Tensor:
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start: Tensor, log_prob: Tensor) -> Tensor:
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x: Tensor, num_classes: int) -> Tensor:
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x: Tensor) -> Tensor:
    return log_x.argmax(1)


def alpha_schedule(
    num_timesteps: int,
    N: int = 100,
    att_1: float = 0.99999,
    att_T: float = 0.000009,
    ctt_1: float = 0.000009,
    ctt_T: float = 0.99999,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # note: 0.0 will tends to raise unexpected behaviour (e.g., log(0.0)), thus avoid 0.0
    assert att_1 > 0.0 and att_T > 0.0 and ctt_1 > 0.0 and ctt_T > 0.0
    assert att_1 + ctt_1 <= 1.0 and att_T + ctt_T <= 1.0

    att = np.arange(0, num_timesteps) / (num_timesteps - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))  # type: ignore
    at = att[1:] / att[:-1]
    ctt = np.arange(0, num_timesteps) / (num_timesteps - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))  # type: ignore
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))  # type: ignore
    ctt = np.concatenate((ctt[1:], [0]))  # type: ignore
    btt = (1 - att - ctt) / N

    def _f(x: np.ndarray) -> Tensor:
        return torch.tensor(x.astype("float64"))

    return _f(at), _f(bt), _f(ct), _f(att), _f(btt), _f(ctt)
