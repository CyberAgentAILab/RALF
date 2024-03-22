from typing import Any

import torch
from torch import Tensor

from .design_seq import reorder


class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # type: ignore
        idx = torch.argmax(x[:, :, 0], -1).unsqueeze(-1)
        output = torch.zeros_like(x[:, :, 0])
        output.scatter_(-1, idx, 1)
        x[:, :, 0] = output
        return x

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class ArgMaxWithReorder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # type: ignore
        idx = torch.argmax(x[:, :, 0], -1).unsqueeze(-1)
        output = torch.zeros_like(x[:, :, 0])
        output.scatter_(-1, idx, 1)
        x[:, :, 0] = output

        # reorder
        i_, j_ = x.shape[:2]
        for i in range(i_):
            for j in range(j_):
                if x[i][j][0][0] == 1:
                    x[i][j][1] = torch.zeros_like(x[i][j][1])

        # x: [bs, max_elem, 2, num_classes]
        for i in range(i_):
            order = reorder(
                x[i, :, 0].detach().cpu(), x[i, :, 1, :4].detach().cpu(), "cxcywh"
            )
            tmp = x[i, :, 1].clone()
            for j in range(j_):
                x[i][j][1] = tmp[int(order[j])]

        return x

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output
