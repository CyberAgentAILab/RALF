from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .positional_encoding import build_position_encoding_1d


class LayoutEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_label: int,
        bbox_type: str = "continuous",
        d_bbox: Optional[int] = 32,
        additional_input: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_label = d_label
        self.bbox_type = bbox_type
        self.additional_input = additional_input
        self.pos_emb = build_position_encoding_1d("layout", d_model)
        self.label_embed = nn.Embedding(d_label, d_model)
        if bbox_type == "continuous":
            self.bbox_embed = nn.Linear(4, d_model)
            if additional_input:
                self.bbox_embed_add = nn.Linear(4, d_model)
        elif bbox_type == "categorical":
            self.bbox_embeds: list[nn.Embedding] = [
                nn.Embedding(d_bbox, d_model) for _ in range(4)  # type: ignore
            ]
            if additional_input:
                self.bbox_embeds_add: list[nn.Embedding] = [
                    nn.Embedding(d_bbox, d_model) for _ in range(4)  # type: ignore
                ]

    def init_weight(self) -> None:
        nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)
        if self.bbox_type == "continuous":
            modules = [self.bbox_embed]
            if self.additional_input:
                modules.append(self.bbox_embed_add)
        elif self.bbox_type == "categorical":
            modules = []
            modules.extend(self.bbox_embeds)  # type: ignore
            if self.additional_input:
                modules.extend(self.bbox_embeds_add)  # type: ignore
        for module in modules:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: dict) -> Any:
        h = self.label_embed(inputs["label"])
        if self.bbox_type == "continuous":
            h += self.bbox_embed(inputs["bbox"])
            if self.additional_input:
                h += self.bbox_embed_add(inputs["pred_bbox"])
        else:
            for i in range(4):
                h += self.bbox_embeds[i](inputs["bbox"][..., i])
        h = self.pos_emb(h)
        return h


class LayoutDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_label: int,
        bbox_type: str = "continuous",
        d_bbox: Optional[int] = 32,
    ) -> None:
        super().__init__()
        self.bbox_type = bbox_type
        self.fc = nn.Linear(d_model, d_model)
        self.fc_label = nn.Linear(d_model, d_label)
        if bbox_type == "continuous":
            self.fc_bbox = nn.Linear(d_model, 4)
        elif bbox_type == "categorical":
            self.fc_bbox = nn.Linear(d_model, 4 * d_bbox)  # type: ignore

    def init_weight(self) -> None:
        for module in [self.fc, self.fc_label, self.fc_bbox]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        return

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = F.relu(self.fc(x))
        outputs = {
            "pred_logits": self.fc_label(h),
        }
        if self.bbox_type == "continuous":
            outputs["pred_boxes"] = torch.sigmoid(self.fc_bbox(h))
        elif self.bbox_type == "categorical":
            raise NotImplementedError
        return outputs


class LayoutRealFakePredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
    ) -> None:
        super().__init__()
        self.fc_label = nn.Linear(d_model, 1)

    def forward(self, h: Tensor) -> Tensor:
        return self.fc_label(h)  # type: ignore
