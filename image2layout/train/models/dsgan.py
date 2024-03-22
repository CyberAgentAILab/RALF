import logging
from typing import Union

import datasets as ds
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .common.base_model import BaseModel
from .common.image import ResnetFeatureExtractor
from .common.retrieval_augment import RetrievalAugmentation
from .common_gan.argmax import ArgMax, ArgMaxWithReorder
from .common_gan.base_model import BaseGANGenerator

logger = logging.getLogger(__name__)


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 32,
        d_model: int = 256,
        num_lstm_layers: int = 4,
    ) -> None:
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=1, padding=1),
        )
        self.lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        logger.info(
            f"Build CNN_LSTM with {in_channels=}, {out_channels=}, {num_lstm_layers=}"
        )
        return

    def forward(self, layout: Tensor, h0: Tensor) -> Tensor:
        """
        Args:
            layout (torch.Tensor): [bs, max_elem, 2, 4]
            h0 (torch.Tensor): [2*lstm_num_layer, bs, c]

        Returns:
            _type_: _description_
        """
        self.lstm.flatten_parameters()
        x = layout.flatten(start_dim=2).permute(0, 2, 1).contiguous()  # [bs, 8, 32]
        x = self.conv(x)  # [bs, 32, 32]
        x = x.permute(0, 2, 1).contiguous()
        output, _ = self.lstm(
            x, (torch.zeros_like(h0).contiguous(), h0.contiguous())
        )  # [bs, 32, 256*2]

        return output


class DSGenerator(BaseGANGenerator):
    def __init__(
        self,
        features: ds.Features,
        d_model: int = 256,
        backbone: str = "resnet50",
        in_channels: int = 8,  # 1dconv: 4 * 2 (xywh, 4class)
        out_channels: int = 32,  # 1dconv: output channel
        num_lstm_layers: int = 4,
        max_seq_length: int = 10,
        apply_weight: bool = False,
        use_reorder: bool = True,
        use_reorder_for_random: bool = False,
    ) -> None:
        num_cls = in_channels // 2
        if num_cls == 4:
            coef: list[float] = [0.8, 1.0, 1.0, 0.1]
        elif num_cls == 5:
            coef: list[float] = [0.8, 0.8, 1.0, 1.0, 0.1]
        assert (
            apply_weight is False
        ), f"{apply_weight=} is not supported. Please read https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/issues/17."
        super(DSGenerator, self).__init__(
            d_model=d_model,
            apply_weight=apply_weight,
            use_reorder=use_reorder,
            use_reorder_for_random=use_reorder_for_random,
            features=features,
            max_seq_length=max_seq_length,
            coef=coef,
        )

        # CNN backbone
        self.encoder = ResnetFeatureExtractor(
            backbone=backbone,
            d_model=d_model,
            num_lstm_layers=4,
            head="lstm",
        )

        # CNN-LSTM
        self.cnnlstm = CNN_LSTM(
            in_channels=in_channels,
            out_channels=out_channels,
            d_model=d_model,
            num_lstm_layers=num_lstm_layers,
        )

        # Predictor
        self.fc1 = nn.Linear(2 * d_model, self.d_label)
        self.fc2 = nn.Linear(2 * d_model, 4)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        h0 = self.encoder(inputs["image"])  # [lstm_num_layer, bs, c]
        layout_feature = inputs["layout"]
        return (h0, layout_feature)

    def decode(self, img_feature: Tensor, layout: Tensor) -> dict[str, Tensor]:
        # LSTM layers
        lstm_output = self.cnnlstm(layout, img_feature)  # [bs, max_elem, max_elem*8*2]
        # HEAD
        cls_label = nn.Softmax(dim=-1)(self.fc1(lstm_output))  # [bs, max_elem, 4]
        box = nn.Sigmoid()(self.fc2(lstm_output))  # [bs, max_elem, 4]
        # Formulate output as dict
        outputs = {"pred_logits": cls_label, "pred_boxes": box.float()}
        return outputs

    def update_per_epoch(
        self, epoch: int, warmup_dis_epoch: int, max_epoch: int
    ) -> None:
        if epoch > warmup_dis_epoch:
            self.adv_weight = 1.0
        else:
            self.adv_weight = 1.0 / warmup_dis_epoch * (epoch - 1)
        logger.info(f"Current {epoch=} {self.adv_weight=}")


class RetrievalAugmentedDSGenerator(DSGenerator):
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
        super(RetrievalAugmentedDSGenerator, self).__init__(**kwargs)

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
        img_feature, layout_feature = super().encode(inputs)
        # img_feature:   # [seq, bs, c]

        img_feature = rearrange(img_feature, "s b c -> b s c")  # [b, seq, c]

        # Retrieval augmentation
        img_feature = self.retrieval_aug(
            image_backbone=self.encoder,
            img_feature=img_feature,
            retrieved_layouts=inputs["retrieved"],
        )  # [b, seq, c]
        img_feature = rearrange(img_feature, "b s c -> s b c")  # [seq, b, c]

        return (img_feature, layout_feature)


class DSDiscriminator(BaseModel):
    LR_MULT: float = 10.0

    def __init__(
        self,
        features: ds.Features,
        backbone="resnet18",
        in_channels=8,
        out_channels=32,
        num_lstm_layers=2,
        d_model: int = 256,
    ):
        super(DSDiscriminator, self).__init__()

        # CNN backbone
        self.encoder = ResnetFeatureExtractor(
            backbone=backbone,
            d_model=d_model,
            num_lstm_layers=num_lstm_layers,
            head="lstm",
        )

        # CNN-LSTM
        self.cnnlstm = CNN_LSTM(
            in_channels=in_channels,
            out_channels=out_channels,
            d_model=d_model,
            num_lstm_layers=num_lstm_layers,
        )

        # Predictor
        self.fc_tf = nn.Linear(2 * d_model, 1)

    def set_argmax(self, use_reorder: bool) -> None:

        if use_reorder:
            self.argmax = ArgMaxWithReorder()  # Differential argmax
        else:
            self.argmax = ArgMax()  # Differential argmax
        return

    def forward(self, img: Tensor, layout: Tensor) -> Tensor:
        h0 = self.encoder(img)
        lstm_output = self.cnnlstm(self.argmax.apply(layout), h0)[:, -1, :]
        tf = self.fc_tf(lstm_output)
        tf = nn.Tanh()(tf)
        return tf
