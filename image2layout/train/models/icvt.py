import copy
import logging
from typing import Any, Callable, Optional, Union

import datasets as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from image2layout.train.global_variables import GEO_KEYS
from image2layout.train.helpers.bucketizer import bucketizer_factory
from image2layout.train.models.common.base_model import BaseModel
from omegaconf import DictConfig
from torch import Tensor

from .common.image import ImageEncoder
from .common.positional_encoding import build_position_encoding_1d
from .common_gan.base_model import BaseGANGenerator

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, num_classes: int, n_boundaries: int = 128) -> None:
        self.bg_idx = num_classes  # note: BG class is not included

        self._bucketizers = {}
        for key in GEO_KEYS:
            self._bucketizers[key] = bucketizer_factory("linear")(
                n_boundaries=n_boundaries
            )

    def __call__(self, inputs: dict) -> Any:
        padding_mask = ~inputs["mask"]
        outputs = {"mask": inputs["mask"]}

        for key in GEO_KEYS:
            outputs[key] = self._bucketizers[key].encode(inputs[key])
            outputs[key][padding_mask] = 0

        if "label" in inputs:
            outputs["label"] = inputs["label"]
            outputs["label"][padding_mask] = self.bg_idx  # BG class

        return outputs

    def decode(self, inputs: dict) -> dict:
        outputs = {"label": inputs["label"]}
        for key in GEO_KEYS:
            outputs[key] = self._bucketizers[key].decode(inputs[key])
        outputs["mask"] = inputs["label"] != self.bg_idx
        return outputs

    def encode(self, inputs: dict) -> dict:
        return self(inputs)


class LayoutDictEncoder(nn.Module):
    def __init__(self, d_model: int, num_classes_w_bg: int, n_boundaries: int) -> None:
        super().__init__()
        self.embed_label = nn.Embedding(num_classes_w_bg, d_model)
        for key in GEO_KEYS:
            setattr(self, f"embed_{key}", nn.Embedding(n_boundaries, d_model))

    def __call__(self, inputs: dict) -> torch.Tensor:
        h = []
        if "label" in inputs:
            h.append(self.embed_label(inputs["label"]))
        for key in GEO_KEYS:
            h.append(getattr(self, f"embed_{key}")(inputs[key]))
        h = torch.cat(h, dim=-1)
        return h


class LayoutDictDecoder(nn.Module):
    def __init__(self, d_model: int, num_classes_w_bg: int, n_boundaries: int) -> None:
        super().__init__()
        self.fc_label = nn.Linear(d_model, num_classes_w_bg)
        for key in GEO_KEYS:
            setattr(self, f"fc_{key}", nn.Linear(d_model, n_boundaries))

    def __call__(self, h: torch.Tensor) -> dict:
        outputs = {}
        outputs["label"] = self.fc_label(h)
        for key in GEO_KEYS:
            outputs[key] = getattr(self, f"fc_{key}")(h)
        return outputs


def _make_grid_like_layout(grid_x: int, grid_y: int) -> Tensor:
    assert grid_x > 0 and grid_y > 0

    cxcy = torch.stack(
        torch.meshgrid(torch.arange(grid_y) / grid_y, torch.arange(grid_x) / grid_x),
        dim=-1,
    )  # (H, W, 2)
    outputs = {"center_y": cxcy[..., 0], "center_x": cxcy[..., 1]}
    outputs["width"] = torch.full((grid_y, grid_x), fill_value=1 / grid_x)
    outputs["height"] = torch.full((grid_y, grid_x), fill_value=1 / grid_y)
    outputs["mask"] = torch.full((grid_y, grid_x), fill_value=True)
    return outputs


class VAEModule(nn.Module):
    def __init__(self, dim_input: int, dim_latent: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(dim_input, dim_latent)
        self.fc_var = nn.Linear(dim_input, dim_latent)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        return {"z": z, "mu": mu, "logvar": logvar}


class ICVTGenerator(BaseModel):
    """
    Geometry Aligned Variational Transformer for Image-conditioned Layout Generation [ACMMM'22]
    https://arxiv.org/abs/2209.00852
    Variational Transformer Networks for Layout Generation
    https://openaccess.thecvf.com/content/CVPR2021/papers/Arroyo_Variational_Transformer_Networks_for_Layout_Generation_CVPR_2021_paper.pdf
    """

    def __init__(
        self,
        features: ds.Features,
        d_model: int = 256,
        backbone: str = "resnet50",
        ga_type: Optional[str] = None,
        kl_mult: float = 1.0,  # adjust loss global scale for kl loss
        decoder_only: bool = False,  # only for debugging
        ignore_bg_bbox_loss: bool = False,  # tried but not effective
        **kwargs: Any,  # for compatibility
    ) -> None:
        super().__init__()
        assert d_model % 4 == 0 and d_model % 5 == 0

        self.d_model = d_model
        self.features = features
        self.kl_mult = kl_mult
        num_classes = features["label"].feature.num_classes

        # TODO: fix these hardcorded values
        self.max_seq_length = 10
        self.num_layers = 6

        self.decoder_only = decoder_only
        self.ignore_bg_bbox_loss = ignore_bg_bbox_loss
        self._learnable_token = nn.Embedding(1, d_model)

        self.n_boundaries = 128
        self.tokenizer = Tokenizer(
            num_classes=num_classes,
            n_boundaries=self.n_boundaries,
        )

        self.layout_encoder = LayoutDictEncoder(
            d_model=d_model // 5,
            num_classes_w_bg=num_classes + 1,
            n_boundaries=self.n_boundaries,
        )
        self.layout_decoder = LayoutDictDecoder(
            d_model=d_model,
            num_classes_w_bg=num_classes + 1,
            n_boundaries=self.n_boundaries,
        )
        self.ga_layout_encoder = LayoutDictEncoder(
            d_model=d_model // 4,
            num_classes_w_bg=num_classes + 1,
            n_boundaries=self.n_boundaries,
        )

        # CNN backbone
        self.encoder = ImageEncoder(
            d_model=d_model,
            backbone_name=backbone,
            num_layers=self.num_layers,
            pos_emb="sine",
        )
        self.pos_emb_1d = build_position_encoding_1d(pos_emb="layout", d_model=d_model)

        self.vae_encoder = GeometryAlignedTransformerDecoder(
            decoder_layer=GeometryAlignedTransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
                ga_type=ga_type,
            ),
            num_layers=self.num_layers,
        )

        self.aap = nn.modules.activation.MultiheadAttention(
            d_model, 8, dropout=0.1, batch_first=True
        )
        self.vae_head = VAEModule(dim_input=d_model, dim_latent=d_model)

        self.vae_decoder = GeometryAlignedTransformerDecoder(
            decoder_layer=GeometryAlignedTransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
                ga_type=ga_type,
            ),
            num_layers=self.num_layers,
        )

        self.loss_label = nn.CrossEntropyLoss()
        self.loss_box = nn.CrossEntropyLoss()
        self.loss_weight_dict = {f"recon_{key}": 1.0 for key in GEO_KEYS}
        self.loss_weight_dict["recon_label"] = 1.0
        self.loss_weight_dict["kl"] = self.kl_mult * 1e-3

        self.init_weights()

    def learnable_token(self, batch_size: int, device: torch.device) -> Tensor:
        dummy_input = torch.zeros((1, 1), device=device, dtype=torch.long)
        token = self._learnable_token(dummy_input)
        token = repeat(token, "1 1 d -> b 1 d", b=batch_size)
        return token

    def init_weights(self) -> None:
        for module in [self.vae_encoder, self.vae_decoder]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> Tensor:
        return self.encoder(inputs["image"])  # [bs, h*w, c]=[128, 330, 256]

    def decode(self):
        raise NotImplementedError

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # don't use this since train/test workflow is very different in vae
        raise NotImplementedError

    def _extract_ga_key_feature(self, inputs: dict) -> Tensor:
        B, _, H, W = inputs["image"].size()
        assert (H, W) == (350, 240)
        layout = _make_grid_like_layout(grid_y=22, grid_x=15)
        layout = self.tokenizer.encode(layout)
        layout = {k: v.view(-1).to(inputs["image"].device) for k, v in layout.items()}
        h = self.ga_layout_encoder(layout)
        h = repeat(h, "x c -> b x c", b=B)
        return h

    def train_loss(
        self,
        inputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        test: bool = False,
    ):
        # get features
        img_feature = self._encode_into_memory(inputs)
        layout_feature = self.layout_encoder(inputs)

        B = img_feature.size(0)
        device = img_feature.device
        ga_input_key = self._extract_ga_key_feature(inputs)

        if self.decoder_only:
            z = self.learnable_token(batch_size=B, device=device)
        else:
            h = self.vae_encoder(
                tgt=layout_feature,
                tgt_key_padding_mask=~inputs["mask"],
                memory=img_feature,
                ga_input_query=layout_feature,
                ga_input_key=ga_input_key,
            )
            pooled_h, _ = self.aap(
                query=self.learnable_token(batch_size=B, device=device),
                key=h,
                value=h,
                key_padding_mask=~inputs["mask"],
            )
            z_dict = self.vae_head(pooled_h)
            z = z_dict["z"]

        # shift input features for training with teacher forcing
        layout_feature_shifted = torch.cat([z, layout_feature[:, :-1]], dim=1)
        layout_feature_shifted = self.pos_emb_1d(layout_feature_shifted)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            layout_feature_shifted.size(1)
        )
        h = self.vae_decoder(
            tgt=layout_feature_shifted,
            memory=img_feature,
            tgt_mask=tgt_mask.to(device),
            ga_input_query=layout_feature_shifted,
            ga_input_key=ga_input_key,
        )  # (B, S, C)

        # pred_logits = self.fc1(h)  # [bs, max_elem, num_classes]
        # pred_boxes = nn.Sigmoid()(self.fc2(h))  # [bs, max_elem, 4]
        outputs = self.layout_decoder(h)

        # Formulate output
        if not self.decoder_only:
            outputs["pred_mu"] = z_dict["mu"]
            outputs["pred_logvar"] = z_dict["logvar"]

        losses = {}
        losses["loss_recon_label"] = self.loss_label(
            rearrange(outputs["label"], "b s c -> b c s"), targets["label"]
        )

        for key in GEO_KEYS:
            if self.ignore_bg_bbox_loss:
                loss_mask = inputs["mask"]
            else:
                loss_mask = torch.full_like(inputs["mask"], fill_value=True)
            losses[f"loss_recon_{key}"] = self.loss_box(
                outputs[key][loss_mask], targets[key][loss_mask]
            )

        if self.decoder_only:
            losses["loss_kl"] = torch.tensor(0.0, device=device)
        else:
            losses["loss_kl"] = -0.5 * torch.mean(
                1
                + outputs["pred_logvar"]
                - outputs["pred_mu"].pow(2)
                - outputs["pred_logvar"].exp()
            )

        for key, value in self.loss_weight_dict.items():
            loss_name = f"loss_{key}"
            losses[loss_name] = value * losses[loss_name]

        return outputs, losses

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Tensor] = None,
        sampling_cfg: Optional[DictConfig] = None,
        return_violation: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = cond["image"].device
        inputs, _ = self.preprocess(
            {
                k: v.to(torch.device("cpu"))
                for k, v in cond.items()
                if torch.is_tensor(v)
            }
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        img_feature = self._encode_into_memory(inputs)
        ga_input_key = self._extract_ga_key_feature(inputs)

        B = img_feature.size(0)
        outputs = {
            "label": torch.zeros((B, 0), dtype=torch.long, device=device),
        }
        for key in GEO_KEYS:
            outputs[key] = torch.zeros((B, 0), dtype=torch.long, device=device)

        if self.decoder_only:
            tgt = self.learnable_token(batch_size=B, device=device)
        else:
            tgt = torch.randn((B, 1, self.d_model), device=device)

        for i in range(self.max_seq_length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
            h = self.vae_decoder(
                tgt=self.pos_emb_1d(tgt),
                memory=img_feature,
                tgt_mask=tgt_mask.to(device),
                ga_input_query=tgt,
                ga_input_key=ga_input_key,
            )
            h = self.layout_decoder(h[:, i : i + 1, :])

            # get most likely label and bbox
            outputs["label"] = torch.cat(
                [
                    outputs["label"],
                    torch.argmax(rearrange(h["label"], "b 1 c -> b c 1"), dim=1),
                ],
                dim=1,
            )
            for key in GEO_KEYS:
                outputs[key] = torch.cat(
                    [
                        outputs[key],
                        torch.argmax(rearrange(h[key], "b 1 c -> b c 1"), dim=1),
                    ],
                    dim=1,
                )
            tgt = torch.cat([tgt, self.layout_encoder(outputs)], dim=1)

        output_seq = self.postprocess(outputs)
        if not return_violation:
            return output_seq
        return output_seq, None

    def update_per_epoch(
        self, epoch: int, warmup_dis_epoch: int, max_epoch: int
    ) -> None:
        # Update beta in ICVT (Eq.10)
        num_cycle = 2
        period = max_epoch // num_cycle
        t = (epoch % period) / period
        if 0.0 <= t < 0.5:
            beta = 0.001
        elif t < 0.75:
            # linearly increase beta
            beta = 0.001 + (0.3 - 0.001) * (t - 0.5) / 0.25
        else:
            beta = 0.3
        self.loss_weight_dict["kl"] = self.kl_mult * beta
        logger.info(f"Current {epoch=} {self.loss_weight_dict['kl']=}")

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:
        tokenized_inputs = self.tokenizer.encode(inputs)
        new_inputs = {
            "image": torch.cat([inputs["image"], inputs["saliency"]], dim=1),
            **tokenized_inputs,
        }
        targets = tokenized_inputs
        return new_inputs, targets

    def postprocess(self, inputs: dict) -> dict:
        inputs = {k: v.to(torch.device("cpu")) for (k, v) in inputs.items()}
        return self.tokenizer.decode(inputs)


class GeometryAlignedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
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
        ga_type: Optional[str] = None,
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
        assert ga_type in ["add", "concat", None]
        self.ga_type = ga_type

        if ga_type == "concat":
            factory_kwargs = {"device": device, "dtype": dtype}
            self.multihead_attn = nn.modules.activation.MultiheadAttention(
                d_model * 2,
                nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs,
            )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        ga_input_query: Optional[Tensor] = None,
        ga_input_key: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                ga_input_query,
                ga_input_key,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    ga_input_query,
                    ga_input_key,
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        ga_input_query: Optional[Tensor] = None,
        ga_input_key: Optional[Tensor] = None,
    ) -> Tensor:
        if self.ga_type == "add":
            query = x + ga_input_query
            key = mem + ga_input_key
            value = mem
        elif self.ga_type == "concat":
            query = torch.cat([x, ga_input_query], dim=-1)
            key = torch.cat([mem, ga_input_key], dim=-1)
            C = mem.size(2)
            value = torch.cat([mem, torch.zeros_like(mem)], dim=-1)
        else:
            # boils down to standard multihead attention
            query, key, value = x, mem, mem

        x = self.multihead_attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        if self.ga_type == "concat":
            # cut off the second half since it is dummy (see above)
            x = x[:, :, :C]
        return self.dropout2(x)


def _get_clones(module, N):
    return nn.modules.container.ModuleList([copy.deepcopy(module) for i in range(N)])


class GeometryAlignedTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        ga_input_query: Optional[Tensor] = None,
        ga_input_key: Optional[Tensor] = None,
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
                ga_input_query=ga_input_query,
                ga_input_key=ga_input_key,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
