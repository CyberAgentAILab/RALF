import logging

import torch
import torch.nn as nn
from image2layout.train.fid.model import load_fidnet_feature_extractor
from image2layout.train.models.common.attention import Attention, FeedForward
from image2layout.train.models.common.positional_encoding import (
    build_position_encoding_1d,
)
from image2layout.train.models.retrieval_augmented_autoreg import (
    extract_retrieved_features,
)
from torch import Tensor

logger = logging.getLogger(__name__)


class RetrievalAugmentation(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        dataset_name: str,
        top_k: int,
        num_classes: int,
        max_seq_length: int,
        use_reference_image: bool,
    ) -> None:
        super().__init__()

        self.top_k = top_k
        self.use_reference_image = use_reference_image

        self.layout_encoder = load_fidnet_feature_extractor(
            dataset_name=dataset_name,
            num_classes=num_classes,
            max_seq_length=max_seq_length,
        )
        self.layout_encoder.enc_transformer.token.requires_grad = False
        for p in self.layout_encoder.parameters():
            p.requires_grad = False

        self.pos_emb_1d = build_position_encoding_1d(
            pos_emb="layout",
            d_model=d_model,
        )
        self.layout_adapter = FeedForward(
            dim=256, hidden_dim=4 * d_model, output_dim=d_model
        )

        self.attn = Attention(d_model, d_model, heads=8, dim_head=64, dropout=0.0)
        self.head = FeedForward(dim=d_model, hidden_dim=4 * d_model, dropout=0.0)

    def preprocess_retrieved_samples(self, retrieved):

        if isinstance(retrieved, list):
            # Called except for training
            assert len(retrieved) == 1
            retrieved = retrieved[0]

        retrieved["image"] = torch.cat(
            [retrieved["image"], retrieved["saliency"]], dim=2
        )
        assert retrieved["image"].size(2) == 4, f"{retrieved['image'].shape=}"

        return retrieved

    def forward(
        self,
        image_backbone,
        img_feature: Tensor,
        retrieved_layouts: dict[str, Tensor],
    ):
        """
        Args:
            img_feature: [bs, hw, d_model]
        """

        # 1. Move to GPU
        retrieved_layouts = {
            k: v.type_as(img_feature)
            for k, v in retrieved_layouts.items()
            if isinstance(v, Tensor)
        }

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=retrieved_layouts,
            top_k=self.top_k,
            image_encoder=image_backbone,
            layout_encoder=self.layout_encoder,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )
        memory_ca = self.attn(
            img_feature, ref_layouts
        )  #### (Optional: Cross-attn or Concat)
        memory = torch.cat([img_feature, memory_ca, ref_layouts], dim=1)
        memory = self.head(memory)
        return memory
