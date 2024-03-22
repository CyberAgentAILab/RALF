import logging
from typing import Optional, Union

import datasets as ds
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .common.base_model import BaseModel
from .common.image import ImageEncoder
from .common.positional_encoding import build_position_encoding_1d
from .common.retrieval_augment import RetrievalAugmentation
from .common_gan.argmax import ArgMax, ArgMaxWithReorder
from .common_gan.base_model import BaseGANGenerator

logger = logging.getLogger(__name__)


class CGLGenerator(BaseGANGenerator):
    def __init__(
        self,
        features: ds.Features,
        backbone: str = "resnet50",
        in_channels: int = 8,  # 1dconv: 4 * 2 (xywh, 4class)
        out_channels: int = 256,  # 1dconv: output channel
        num_layers: int = 6,
        max_seq_length: int = 10,
        apply_weight: bool = True,
        d_model: int = 256,
        use_reorder: bool = False,
        use_reorder_for_random: bool = False,
        auxilary_task: Optional[str] = None,
    ) -> None:
        coef: list[float] = [1.0] * (in_channels // 2)
        super(CGLGenerator, self).__init__(
            d_model=d_model,
            apply_weight=apply_weight,
            use_reorder=use_reorder,
            use_reorder_for_random=use_reorder_for_random,
            features=features,
            max_seq_length=max_seq_length,
            coef=coef,
            auxilary_task=auxilary_task,
        )

        # CNN backbone
        self.encoder = ImageEncoder(
            d_model=d_model,
            backbone_name=backbone,
            num_layers=num_layers,
            pos_emb="sine",
        )
        self.layout_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=1, padding=1),
        )
        self.pos_emb_1d = build_position_encoding_1d(
            pos_emb="layout", d_model=out_channels
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=out_channels,
                nhead=8,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        # Predictor
        self.head = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(out_channels, self.d_label, bias=False)
        self.fc2 = nn.Linear(out_channels, 4, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        for module in [self.encoder.transformer_encoder, self.transformer_decoder]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        # Enceder image
        h0 = self.encoder(inputs["image"])  # [bs, h*w, c]=[128, 330, 256]

        layout = (
            inputs["layout"].flatten(start_dim=2).permute(0, 2, 1).contiguous()
        )  # [bs, c, elem]
        layout_output = self.layout_encoder(layout)  # [bs, elem, 256]
        layout_output = rearrange(layout_output, "b c n -> b n c")
        layout_output = self.pos_emb_1d(layout_output)  # [bs, elem, 256]

        return (h0, layout_output)

    def decode(self, img_feature: Tensor, layout_feature: Tensor) -> dict[str, Tensor]:

        # Transformer Decoder
        dec = self.transformer_decoder(
            tgt=layout_feature, memory=img_feature
        )  # [bs, 10, 256]
        # Head
        cls_label = self.fc1(dec)  # [bs, max_elem, num_classes]
        box = nn.Sigmoid()(self.fc2(dec))  # [bs, max_elem, 4]
        # Formulate output
        outputs = {"pred_logits": cls_label, "pred_boxes": box}
        return outputs

    def update_per_epoch(
        self, epoch: int, warmup_dis_epoch: int, max_epoch: int
    ) -> None:
        if epoch < warmup_dis_epoch:
            self.adv_weight = 0.0
        elif epoch <= max_epoch:
            self.adv_weight = (epoch - warmup_dis_epoch) / (
                max_epoch - warmup_dis_epoch
            )
        else:
            self.adv_weight = 1.0
        logger.info(f"Current {epoch=} {self.adv_weight=}")


class RetrievalAugmentedCGLGenerator(CGLGenerator):
    def __init__(
        self,
        db_dataset,
        top_k: int,
        dataset_name: str,
        retrieval_backbone: str,
        random_retrieval: bool,
        saliency_k: Union[int, str],
        use_reference_image: bool,
        **kwargs,
    ) -> None:
        super(RetrievalAugmentedCGLGenerator, self).__init__(**kwargs)

        self.top_k = top_k
        self.random_retrieval = random_retrieval

        self.retrieval_aug = RetrievalAugmentation(
            d_model=self.d_model,
            top_k=top_k,
            dataset_name=dataset_name,
            num_classes=self.num_classes,
            max_seq_length=self.max_seq_length,
            use_reference_image=use_reference_image,
        )

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:

        new_inputs, targets = super().preprocess(inputs)

        retrieved_samples = self.retrieval_aug.preprocess_retrieved_samples(
            inputs["retrieved"]
        )
        new_inputs["retrieved"] = retrieved_samples

        return new_inputs, targets

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        img_feature, layout_feature = super()._encode_into_memory(inputs)

        # Retrieval augmentation
        img_feature = self.retrieval_aug(
            image_backbone=self.encoder,
            img_feature=img_feature,
            retrieved_layouts=inputs["retrieved"],
        )  # [b, seq, c]

        return (img_feature, layout_feature)


class CGLDiscriminator(BaseModel):
    LR_MULT: float = 10.0

    def __init__(
        self,
        features: ds.Features,
        backbone: str = "resnet18",
        in_channels: int = 8,
        out_channels: int = 256,
        num_layers: int = 4,
        d_model: int = 256,
        max_seq_length: int = 10,
    ):
        super(CGLDiscriminator, self).__init__()

        # CNN backbone
        self.encoder = ImageEncoder(
            d_model=d_model,
            backbone_name=backbone,
            num_layers=num_layers,
            pos_emb="sine",
        )
        self.layout_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=1, padding=1),
        )
        self.pos_emb_1d = build_position_encoding_1d(
            pos_emb="layout", d_model=out_channels
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=out_channels,
                nhead=8,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
            ),
            num_layers=num_layers,
        )  # type: ignore

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * max_seq_length),
            nn.Linear(d_model * max_seq_length, 1, bias=False),
        )

        self.argmax = None

        self.init_weights()

    def init_weights(self) -> None:
        for module in [self.encoder.transformer_encoder, self.transformer_decoder]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def set_argmax(self, use_reorder: bool) -> None:
        if self.argmax is not None:
            return

        if use_reorder:
            self.argmax = ArgMaxWithReorder()  # Differential argmax  # type: ignore
        else:
            self.argmax = ArgMax()  # Differential argmax  # type: ignore
        return

    def forward(self, img: Tensor, layout: Tensor) -> Tensor:  # type: ignore
        h0 = self.encoder(img)  # [lstm_num_layer, bs, c]

        layout = (
            self.argmax.apply(layout).flatten(start_dim=2).permute(0, 2, 1).contiguous()
        )  # [bs, c, elem]
        layout_output = self.layout_encoder(layout)  # [bs, elem, 256]
        layout_output = rearrange(layout_output, "b c n -> b n c")
        layout_output = self.pos_emb_1d(layout_output)  # [bs, elem, 256]
        dec = self.transformer_decoder(tgt=layout_output, memory=h0)  # [bs, 10, 256]
        # dec = dec.mean(dim=1)  # TODO: correct way?
        dec = rearrange(dec, "b n c -> b (n c)")
        tf = self.head(dec)
        tf = nn.Tanh()(tf)  # [bs, 10, 1]
        return tf  # type: ignore
