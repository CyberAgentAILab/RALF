import logging
import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from image2layout.train.models.common.positional_encoding import (
    build_position_encoding_1d,
)
from torch import Tensor

logger = logging.getLogger(__name__)

TIMESTEP_TYPES = [
    None,
    "adalayernorm",
    "adainnorm",
    "adalayernorm_abs",
    "adainnorm_abs",
    "adalayernorm_mlp",
    "adainnorm_mlp",
]


# https://github.com/microsoft/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/transformer_utils.py
class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000) -> None:
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor) -> Tensor:
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _AdaNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ) -> None:
        super().__init__()
        if "abs" in emb_type:
            self.emb: nn.Module = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ) -> None:
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ) -> None:
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
        return x


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Overriding nn.TransformerDecoderLayer to take timestep embedding for diffusion-like models
    """

    def __init__(  # type: ignore
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        timestep_type: Optional[str] = None,
        max_timestep: int = 100,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        assert timestep_type in TIMESTEP_TYPES
        # time-adaptive norm. is applied except norm3
        # https://github.com/microsoft/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/transformer_utils.py#L263-L267
        if timestep_type is not None:
            if "adalayernorm" in timestep_type:
                self.norm1 = AdaLayerNorm(d_model, max_timestep, timestep_type)  # type: ignore
                self.norm2 = AdaLayerNorm(d_model, max_timestep, timestep_type)  # type: ignore
            elif "adainnorm" in timestep_type:
                self.norm1 = AdaInsNorm(d_model, max_timestep, timestep_type)  # type: ignore
                self.norm2 = AdaInsNorm(d_model, max_timestep, timestep_type)  # type: ignore
            else:
                raise NotImplementedError
        self.timestep_type = timestep_type

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            if self.timestep_type is not None:
                h = self.norm1(x, timestep)
            else:
                h = self.norm1(x)
            x = x + self._sa_block(h, tgt_mask, tgt_key_padding_mask)

            if self.timestep_type is not None:
                h = self.norm2(x, timestep)
            else:
                h = self.norm2(x)
            x = x + self._mha_block(h, memory, memory_mask, memory_key_padding_mask)

            x = x + self._ff_block(self.norm3(x))
        else:
            h = x + self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            if self.timestep_type is not None:
                x = self.norm1(h, timestep)
            else:
                x = self.norm1(h)

            h = x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            if self.timestep_type is not None:
                x = self.norm2(h, timestep)
            else:
                x = self.norm2(h)

            x = self.norm3(x + self._ff_block(x))

        return x


class CustomTransformerDecoder(nn.TransformerDecoder):
    """
    Overriding nn.TransformerDecoder to take timestep embedding for diffusion-like models
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        decoder_layer: CustomTransformerDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)  # type: ignore

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                timestep=timestep,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class DiscreteDiffusionDecoder(nn.Module):
    def __init__(
        self,
        d_label: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        timestep_type="adalayernorm",
        max_timestep: int = 100,
        pos_emb: str = "layout",
        **kwargs,
    ) -> None:
        super().__init__()
        self.pos_emb = build_position_encoding_1d(pos_emb=pos_emb, d_model=d_model)

        self.transformer_decoder = CustomTransformerDecoder(
            # note: employ Pre-LN for its stability
            decoder_layer=CustomTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                norm_first=True,
                max_timestep=max_timestep,
                timestep_type=timestep_type,
                # dropout=0.0,  # used in layoutdm
            ),
            num_layers=num_layers,
        )
        self.emb = nn.Embedding(d_label, d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_label, bias=False)
        )

    def init_weight(self) -> None:
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for module in self.head:
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        timestep: Optional[Tensor] = None,
        # below two are dummy args (for compatibility with BaseDecoder)
        tgt_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        h = self.emb(tgt)
        h = self.pos_emb(h)

        h = self.transformer_decoder(tgt=h, memory=memory, timestep=timestep)

        h = self.head(h)
        return h


if __name__ == "__main__":
    B, S, C = 16, 5, 128
    decoder_layer = CustomTransformerDecoderLayer(
        d_model=C, nhead=8, batch_first=True, timestep_type="adalayernorm"
    )
    decoder = CustomTransformerDecoder(decoder_layer=decoder_layer, num_layers=4)

    timestep = torch.randint(low=0, high=100, size=(B,))

    tgt = torch.rand(B, S, C)
    memory = torch.rand(B, 3, C)
    out = decoder(tgt, memory, timestep=timestep)
