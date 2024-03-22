import logging
import os

import fsspec
import torch
import torch.nn as nn
from image2layout.train.global_variables import PRECOMPUTED_WEIGHT_DIR
from torch import BoolTensor, LongTensor, Tensor

from .data import BBOX_KEYS

logger = logging.getLogger(__name__)


class TransformerWithToken(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int
    ) -> None:
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer("token_mask", token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )  # type: ignore

    def forward(self, x: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        # x: [N, B, E]
        # padding_mask: [B, N]
        #   `False` for valid values
        #   `True` for padded values

        B = x.size(1)

        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x


class FIDNetV3(nn.Module):
    def __init__(
        self,
        num_label: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        max_bbox: int = 50,
    ) -> None:
        super().__init__()

        # encoder
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(
            d_model=d_model,
            dim_feedforward=d_model // 2,
            nhead=nhead,
            num_layers=num_layers,
        )

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model // 2
        )
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)  # type: ignore

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def _preprocess(self, inputs: dict) -> tuple[Tensor, LongTensor, BoolTensor]:
        padding_mask = ~inputs["mask"]
        bbox = torch.stack([inputs[key] for key in BBOX_KEYS], dim=-1)
        return bbox, inputs["label"], padding_mask

    def extract_features(self, inputs: dict[str, Tensor]) -> Tensor:
        # due to the compatibility with other generator mdoels, interface is changed.
        bbox, label, padding_mask = self._preprocess(inputs)
        h_bbox = self.fc_bbox(bbox)
        h_label = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([h_bbox, h_label], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]  # type: ignore

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """
        Note: encoder does not use positional embedding,
              and the extracted feat. is invariant w.r.t. element order
        """
        B, N = inputs["label"].size()
        x = self.extract_features(inputs)

        logit_disc = self.fc_out_disc(x).squeeze(-1)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.dec_fc_in(x))

        x = self.dec_transformer(x, src_key_padding_mask=~inputs["mask"])
        # x = x.permute(1, 0, 2)[~padding_mask]
        x = x.permute(1, 0, 2)

        # logit_cls: [B, N, L]    bbox_pred: [B, N, 4]
        logit_cls = self.fc_out_cls(x)
        bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

        return logit_disc, logit_cls, bbox_pred


def load_fidnet_v3(model: FIDNetV3, ckpt_dir: str) -> FIDNetV3:
    ckpt_path = os.path.join(ckpt_dir, "model_best.pth.tar")
    fs, path_prefix = fsspec.core.url_to_fs(ckpt_path)

    if fs.exists(path_prefix):
        weight_path = path_prefix
    else:
        weight_path = os.path.join(
            PRECOMPUTED_WEIGHT_DIR, "fidnet", *path_prefix.split("/")[-2:]
        )

    logger.info(f"Loading FIDNetV3 ({weight_path=}) ...")
    with fsspec.open(weight_path, "rb") as file_obj:
        x = torch.load(file_obj, map_location=torch.device("cpu"))
    _ = model.load_state_dict(x["state_dict"])
    model.eval()
    return model


def load_fidnet_feature_extractor(
    dataset_name: str = "",
    num_classes: int = 3,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    max_seq_length: int = 10,
    ckpt_dir="tmp/fidnet",
) -> nn.Module:
    model = FIDNetV3(
        num_label=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_bbox=max_seq_length,
    )
    if dataset_name == "pku":
        dataset_name = "pku10"
    ckpt_dir = os.path.join(ckpt_dir, dataset_name)
    model = load_fidnet_v3(model, ckpt_dir)
    del model.pos_token
    del model.dec_transformer
    del model.fc_out_disc
    del model.fc_out_cls
    del model.fc_out_bbox
    return model
