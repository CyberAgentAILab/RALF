import logging
import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import LongTensor, Tensor

logger = logging.getLogger(__name__)


def build_position_encoding_1d(
    pos_emb: str = "layout", d_model: int = 256, **kwargs: Any
) -> nn.Module:  # type: ignore
    if pos_emb == "layout":
        position_embedding: nn.Module = PositionalEncoding1d(d_model=d_model, **kwargs)
    elif pos_emb == "elem_attr":
        position_embedding = ElemAttrPositionalEncoding1d(d_model=d_model, **kwargs)
    elif pos_emb == "none":
        position_embedding = nn.Identity()
    else:
        raise ValueError(f"not supported {pos_emb}")

    return position_embedding


def build_position_encoding_2d(
    pos_emb: str = "sine",
    d_model: int = 256,
    **kwargs: Any,
) -> nn.Module:
    """
    The function takes a feature map (B, C, H', W') and returns a feature map (B, H'*W', C)
    """
    if pos_emb == "reshape" or pos_emb == "none":
        position_embedding = ImageReshaper(d_model=d_model)
    elif pos_emb == "sine":
        position_embedding = PositionEmbeddingSine(d_model=d_model, normalize=True)
    elif pos_emb == "learnable":
        position_embedding = PositionEmbeddingLearned(d_model=d_model)
    else:
        raise ValueError(f"not supported {pos_emb}")

    return position_embedding


class ImageReshaper(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(1) == self.d_model, f"{x.size(1)} != {self.d_model}"
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class PositionalEncoding1d(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        # max_len: int = 10000,
        max_len: int = 5000,
        batch_first: bool = True,
        scale_input: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.scale_input = scale_input

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: LongTensor) -> Tensor:
        """
        Args:
            x: Tensor, shape :
                [seq_len, batch_size, embedding_dim] (if batch_first)
                [batch_size, seq_len, embedding_dim] (else)
        """
        h = x * math.sqrt(self.d_model) if self.scale_input else x
        if self.batch_first:
            S = h.size(1)
            h = h + self.pe[:, :S]
        else:
            S = x.size(0)
            h = h + self.pe[:S]
        return self.dropout(h)  # type: ignore


class ElemAttrPositionalEncoding1d(nn.Module):
    """
    Note: assuming batch_first
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        scale_input: bool = True,
        n_attr_per_elem: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.scale_input = scale_input
        self.n_attr_per_elem = n_attr_per_elem

        self.attr_embed = nn.Embedding(n_attr_per_elem, d_model // 2)
        self.elem_embed = nn.Embedding(max_len // n_attr_per_elem, d_model // 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.attr_embed.weight)
        nn.init.uniform_(self.elem_embed.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape :
                [seq_len, batch_size, embedding_dim] (if batch_first)
        """
        h = x * math.sqrt(self.d_model) if self.scale_input else x
        B, S, _ = h.size()
        assert S % self.n_attr_per_elem == 0
        indices = torch.arange(S).to(x.device)

        # (1, 2, 3) -> (1, ..., 1, 2, ..., 2, 3, ..., 3, ...)
        attr_indices = indices % self.n_attr_per_elem
        attr_indices = repeat(attr_indices, "s -> b s", b=B)
        attr_pe = self.attr_embed(attr_indices)

        # (1, 2, 3) -> (1, 2, 3, 1, 2, 3, ...)
        elem_indices = torch.div(indices, self.n_attr_per_elem, rounding_mode="floor")
        elem_indices = repeat(elem_indices, "s -> b s", b=B)
        elem_pe = self.elem_embed(elem_indices)

        h = h + torch.cat([attr_pe, elem_pe], dim=-1)
        return self.dropout(h)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, d_model=64, temperature=10000, normalize=False, scale=None
    ) -> None:
        super().__init__()
        self.d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.reshape = ImageReshaper(d_model)

    def forward(self, input: LongTensor) -> Tensor:
        bs, c, h, w = input.size()  # [bs, h*w, c]
        y, x = torch.meshgrid(
            torch.arange(h).type_as(input),
            torch.arange(w).type_as(input),
            indexing="ij",
        )

        if self.normalize:
            y = y / (h - 1)  # Normalize y coordinates to [0, 1]
            x = x / (w - 1)  # Normalize x coordinates to [0, 1]
            y = y * self.scale
            x = x * self.scale
        dim_t = torch.arange(self.d_model).type_as(input)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.d_model
        )

        pos_x = x.flatten()[None, :, None] / dim_t
        pos_y = y.flatten()[None, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).repeat(bs, 1, 1)

        output = self.reshape(input) + pos

        return output


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, d_model // 2)
        self.col_embed = nn.Embedding(50, d_model // 2)
        self.reset_parameters()
        self.reshape = ImageReshaper(d_model)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, input):
        bs, c, h, w = input.size()  # [bs, c, h, w]
        i = torch.arange(w, device=input.device)
        j = torch.arange(h, device=input.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(bs, 1, 1, 1)
        )

        output = input + pos
        output = self.reshape(output)

        return output
