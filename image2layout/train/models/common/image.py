import logging
import os

import fsspec
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from image2layout.train.global_variables import PRECOMPUTED_WEIGHT_DIR
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor

from .positional_encoding import build_position_encoding_2d

logger = logging.getLogger(__name__)

NORMALIZE = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
RESNET_WEIGHT = {
    "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
}

class ResnetBackbone(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        d_model: int = 256,
        num_lstm_layers: int = 4,
        head: str = "lstm",
    ) -> None:
        # CNN backbone
        super(ResnetBackbone, self).__init__()

        if backbone == "resnet50":
            resnet = timm.create_model("resnet50")
            cache_path = "resnet50_a1_0-14fe96d1.pth"
            fs, path_prefix = fsspec.core.url_to_fs(cache_path)
            if not fs.exists(path_prefix):
                cache_path = os.path.join(PRECOMPUTED_WEIGHT_DIR, cache_path)
                fs, path_prefix = fsspec.core.url_to_fs(cache_path)
                assert fs.exists(path_prefix), f"{path_prefix} does not exist"
            with fs.open(path_prefix, "rb") as f:
                resnet_weight = torch.load(f)
            log = resnet.load_state_dict(resnet_weight)
            ch = [1024, 2048]
        else:
            resnet = timm.create_model("resnet18")
            cache_path = "resnet18-5c106cde.pth"
            fs, path_prefix = fsspec.core.url_to_fs(cache_path)
            if not fs.exists(path_prefix):
                cache_path = os.path.join(PRECOMPUTED_WEIGHT_DIR, cache_path)
                fs, path_prefix = fsspec.core.url_to_fs(cache_path)
                assert fs.exists(path_prefix), f"{path_prefix} does not exist. Please run 'poetry run gdown --id 1cT_FB6PlcgPysJeYsXJ5rdiy96eF7hSm -O ./cache/PRECOMPUTED_WEIGHT_DIR' to download the weight file."
            with fs.open(path_prefix, "rb") as f:
                resnet_weight = torch.load(f)
            log = resnet.load_state_dict(resnet_weight)
            ch = [256, 512]
        print(f"Load {backbone}: {log}")

        return_nodes = {"layer4": "layer4", "layer3": "layer3"}
        self.body = create_feature_extractor(resnet, return_nodes=return_nodes)

        # make the first conv to have four channels
        params = {
            key: getattr(self.body.conv1, key)
            for key in ["kernel_size", "stride", "padding", "out_channels"]
        }
        weight = self.body.conv1.weight.data
        weight = torch.cat([weight, torch.mean(weight, dim=1, keepdim=True)], dim=1)
        self.body.conv1 = nn.Conv2d(in_channels=4, bias=False, **params)
        self.body.conv1.weight.data = weight

        # FPN
        self.fpn_conv11_4 = nn.Conv2d(ch[0], 256, 1, 1, 0)
        self.fpn_conv11_5 = nn.Conv2d(ch[1], 256, 1, 1, 0)
        self.fpn_conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.proj = nn.Conv2d(512, d_model, 1, 1, 0)

        assert head in ["lstm", "transformer"]
        self.head = head
        if head == "lstm":
            self.fc_h0 = nn.Linear(330, num_lstm_layers * 2)  # 330=22*15

    def forward(self, img: Tensor) -> Tensor:  # type: ignore
        """
        Args:
            img (torch.Tensor): [bs, 4, H, W], [0, 1]

        Returns:
            h0 (torch.Tensor): []
        """
        # Multi-sacle feature
        h = self.body(img)
        resnet_f4 = h["layer3"]
        resnet_f5 = h["layer4"]

        resnet_f4p = self.fpn_conv11_4(resnet_f4)
        resnet_f5p = self.fpn_conv11_5(resnet_f5)
        resnet_f5up = F.interpolate(
            resnet_f5p, size=resnet_f4p.shape[2:], mode="nearest"
        )
        resnet_fused = torch.concat(
            [resnet_f5up, self.fpn_conv33(resnet_f5up + resnet_f4p)], dim=1
        )
        resnet_proj = self.proj(resnet_fused)  # [bs, c, h, w]

        if self.head == "lstm":
            resnet_flat = resnet_proj.flatten(start_dim=-2)  # [bs, c, h*w]
            h0 = self.fc_h0(resnet_flat)  # [bs, c, 2*lstm_num_layer]
            h0 = h0.permute(2, 0, 1)  # [2*lstm_num_layer, bs, c]=[8, 128, 256]
            return h0  # type: ignore

        elif self.head == "transformer":
            return resnet_proj  # [bs, c, h, w]  # type: ignore


class ResnetFeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.extractor = ResnetBackbone(**kwargs)

    def forward(self, *args):
        return self.extractor(*args)


class ImageEncoder(nn.Module):
    """
    This design follows encoder part of CGL-GAN (https://arxiv.org/abs/2205.00303)
    (i) extract image features
    (ii) flatten them into 1D sequence
    (iii) consider interaction using standard Transformer Encoder
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        backbone_name: str = "resnet50",
        num_layers: int = 6,
        dropout: float = 0.1,
        pos_emb: str = "sine",
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        self.extractor = ImageFeatureExtractor(
            d_model=d_model, backbone_name=backbone_name
        )
        logger.info(f"Build ImageEncoder with {pos_emb=}, {d_model=}")
        self.pos_emb = build_position_encoding_2d(pos_emb, d_model)
        self.transformer_encoder = nn.TransformerEncoder(  # type: ignore
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=dropout,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

    def init_weight(self) -> None:
        self.extractor.init_weight()
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return

    def forward(self, image: Tensor) -> Tensor:
        h = self.extractor(image.clone())
        h = self.pos_emb(h)
        h = self.transformer_encoder(h)
        return h  # type: ignore


class ImageFeatureExtractor(nn.Module):
    """
    This design follows encoder part of CGL-GAN (https://arxiv.org/abs/2205.00303)
    """

    def __init__(self, d_model: int = 256, backbone_name: str = "resnet18") -> None:
        super().__init__()
        return_nodes = {"layer4": "layer4", "layer3": "layer3"}
        model = getattr(torchvision.models, backbone_name)(
            weights=RESNET_WEIGHT[backbone_name]
        )
        self.body = create_feature_extractor(model, return_nodes=return_nodes)
        num_channels_dict = {
            "resnet18": {"layer3": 256, "layer4": 512},
            "resnet50": {"layer3": 1024, "layer4": 2048},
        }

        self.conv11 = nn.Conv2d(
            num_channels_dict[backbone_name]["layer4"], d_model // 2, 1
        )
        self.conv22 = nn.Conv2d(
            num_channels_dict[backbone_name]["layer3"], d_model // 2, 1
        )
        self.conv33 = nn.Conv2d(d_model // 2, d_model // 2, 1)

        # make the first conv to have four channels
        params = {
            key: getattr(self.body.conv1, key)
            for key in ["kernel_size", "stride", "padding", "out_channels"]
        }
        weight = self.body.conv1.weight.data
        weight = torch.cat([weight, torch.mean(weight, dim=1, keepdim=True)], dim=1)
        self.body.conv1 = nn.Conv2d(in_channels=4, bias=False, **params)
        self.body.conv1.weight.data = weight

    def init_weight(self) -> None:
        for conv in [self.conv11, self.conv22, self.conv33]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        x[:, 0:3] = NORMALIZE(x[:, 0:3])
        h = self.body(x)
        l3, l4 = h["layer3"], h["layer4"]
        f_up = F.interpolate(self.conv11(l4), l3.size()[2:], mode="bilinear")
        h = torch.cat(
            [f_up, self.conv33(f_up + self.conv22(l3))], dim=1
        )  # [b, c, h, w]
        # Comment out because sine positional embedding needs spatial information
        # h = rearrange(h, "b c h w -> b (h w) c")
        return h  # type: ignore
