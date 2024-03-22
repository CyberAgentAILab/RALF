import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import BoolTensor, Tensor

from .positional_encoding import build_position_encoding_1d

logger = logging.getLogger(__name__)


class BaseDecoder(nn.Module):
    def __init__(
        self,
        d_label: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        tie_weights: bool = False,
        pos_emb: str = "layout",
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        self.tie_weights = tie_weights
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )
        self.d_model = d_model
        self.emb = nn.Embedding(d_label, d_model)
        self.pos_emb = build_position_encoding_1d(pos_emb, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_label, bias=False)
        )

        # Weight tying
        # - https://arxiv.org/abs/1608.05859
        # - https://arxiv.org/abs/1611.01462
        if self.tie_weights:
            raise NotImplementedError
            # self.head.weight = self.emb.weight

        self.use_paramter_ablation = False
        if d_model != 256:
            self.use_paramter_ablation = True

            self.memory_dim_converter = nn.Linear(
                256,
                d_model,
                bias=False)
        
        n_params = sum([p.numel() for p in self.parameters()]) / 1e6
        logger.info(
            f"BaseDecoder: {pos_emb=}, {d_model=}, {d_label=}, {num_layers=}, {nhead=}, {n_params=}, {self.use_paramter_ablation=}"
        )

    def reset_embedding_layer(self, d_in):
        prev_in_emb = self.emb.num_embeddings
        self.emb = nn.Embedding(d_in, self.d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        logger.info(
            f"Update embedding layer in {type(self).__name__}: from {prev_in_emb} to {d_in}"
        )

    def init_weight(self) -> None:
        for p in self.transformer.parameters():
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
        tgt_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        soft_token: Optional[Tensor] = None,
        soft_token_mask: Optional[BoolTensor] = None,
        emb_decoder_token: Optional[Tensor] = None,
        emb_soft_token: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Variable names are following nn.TransformerDecoder to avoid confusion.
        """
        h = self.emb(tgt)  # (B, S, D)
        h = self.pos_emb(h)  # [B, 50]

        if self.use_paramter_ablation:
            memory = self.memory_dim_converter(memory)

        if emb_decoder_token is not None or emb_soft_token is not None:
            assert emb_decoder_token is not None and emb_soft_token is not None
            h = h + emb_decoder_token
            soft_token = soft_token + emb_soft_token

        if soft_token is not None:
            reject_index = soft_token.size(1)
            h = torch.cat([soft_token, h], dim=1)
            tgt_key_padding_mask = torch.cat(
                [soft_token_mask, tgt_key_padding_mask], dim=1
            )

        if is_causal:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(h.size(1))
            h = self.transformer(
                h,
                memory,
                tgt_mask=tgt_mask.to(h.device),  # TODO: check if it is really causal
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  # TODO: use tgt_is_causal (pt2.0 ~ ?) to ease the process
        else:
            h = self.transformer(
                h,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        if soft_token is not None:
            h = h[:, reject_index:, :]

        h = self.head(h)

        return h

    # def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
    #     for name, param in self.named_parameters(recurse=recurse):
    #         yield param

    # overide this for weight tying
    # def _named_members(
    #     self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    # ):
    #     r"""Helper method for yielding various names + members of modules."""
    #     memo = set()
    #     modules = (
    #         self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
    #         if recurse
    #         else [(prefix, self)]
    #     )
    #     for module_prefix, module in modules:
    #         members = get_members_fn(module)
    #         for k, v in members:
    #             if v is None or v in memo:
    #                 continue
    #             if remove_duplicate:
    #                 memo.add(v)
    #             name = module_prefix + ("." if module_prefix else "") + k

    #             if self.tie_weights and name == "decoder.head.weight":
    #                 continue

    #             yield name, v


class SeqLengthDistribution(nn.Module):
    def __init__(self, max_seq_length: int, weight: float = 0.999) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.weight = weight

        # logger.warning("EMA for seq_length is computed during training")
        fill_value = 1 / max_seq_length
        self.register_buffer(
            "n_elements_prob",
            torch.full((max_seq_length,), fill_value=fill_value),
        )

    def __call__(self) -> None:
        raise NotImplementedError

    def update(self, mask: BoolTensor) -> None:
        assert mask.ndim == 2  # (B, S)
        N = self.max_seq_length
        batch_prob = mask.sum(dim=1).bincount(minlength=N + 1)[1:] / mask.size(0)
        self.n_elements_prob = self.weight * self.n_elements_prob
        self.n_elements_prob += (1.0 - self.weight) * batch_prob.to(
            self.n_elements_prob
        )

    def sample(self, batch_size: int) -> Tensor:
        n_elements = torch.multinomial(
            self.n_elements_prob.cpu(), batch_size, replacement=True
        )
        n_elements += 1  # shoule be in range [1, cfg.dataset.max_seq_length]
        return n_elements


class UserConstraintTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_label: int,
        embedding_layer: Optional[nn.Module],
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(  # type: ignore
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )
        if embedding_layer is not None:
            self.emb = embedding_layer
        else:
            self.emb = nn.Embedding(d_label, d_model)
            nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.pos_emb = build_position_encoding_1d("layout", d_model)

        logger.info(
            f"{self.__class__.__name__}: {d_label=}, {d_model=}, {num_layers=}, {nhead=}"
        )

    def init_weight(self) -> None:
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor,
        task_token: Optional[Tensor],
    ) -> Tensor:
        h = self.emb(src)
        h = self.pos_emb(h)
        h = self.encoder(src=h, src_key_padding_mask=src_key_padding_mask)

        if task_token is not None:
            task_token = self.emb(task_token)
            h = h + task_token

        return h
