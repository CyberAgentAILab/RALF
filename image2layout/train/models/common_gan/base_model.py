import logging
import random
from abc import abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from image2layout.train.helpers.layout_tokenizer import GEO_KEYS
from image2layout.train.helpers.task import COND_TYPES
from image2layout.train.models.common.base_model import BaseModel
from omegaconf import DictConfig
from torch import Tensor

from .layout_initializer import preprocess_layout, random_init_layout
from .rec_loss import HungarianMatcher, SetCriterion

logger = logging.getLogger(__name__)


class BaseGANGenerator(BaseModel):
    def __init__(
        self,
        *,
        d_model,
        apply_weight,
        use_reorder,
        use_reorder_for_random,
        features,
        max_seq_length,
        coef,
        auxilary_task: Optional[str] = "uncond",
    ) -> None:
        super(BaseGANGenerator, self).__init__()

        self.d_model = d_model
        self.apply_weight = apply_weight
        self.use_reorder = use_reorder
        self.use_reorder_for_random = use_reorder_for_random
        self.max_seq_length = max_seq_length
        self.features = features
        self.coef = coef
        self.auxilary_task = auxilary_task
        logger.info(f"In GAN backbone, {auxilary_task=}")
        assert (
            auxilary_task in COND_TYPES
        ), f"{auxilary_task=} must be one of {COND_TYPES}"
        # if auxilary_task is not None or auxilary_task != "uncond":
        #     assert not use_reorder and not use_reorder_for_random

        self.num_classes = num_classes = self.features["label"].feature.num_classes
        self.d_label = num_classes + 1  # +1 for no-object

        matcher = HungarianMatcher(2.0, 5.0, 2.0)
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }
        self.criterion_rec = SetCriterion(
            num_classes, matcher, weight_dict, self.coef, ["labels", "boxes"]
        )  # type: ignore
        self.criterion_adv = nn.HingeEmbeddingLoss()

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:
        """
        Args:
            inputs (dict):
                id
                image_width
                image_height
                image
                saliency[]
                label
                center_x
                center_y
                width
                height
                mask

        Returns:
            tuple[dict, dict]: _description_
        """
        batch = preprocess_layout(
            inputs,
            max_elem=self.max_seq_length,
            num_classes=self.d_label,
            use_reorder=self.use_reorder,
        )
        random_layout: Tensor = random_init_layout(
            batch_size=batch["image"].size(0),
            seq_length=self.max_seq_length,
            coef=self.coef,
            use_reorder=self.use_reorder_for_random,
            num_classes=self.d_label,
        ).type_as(
            batch["image"]
        )  # [bs, 10, 2, NUM_ELEM]

        label_gt: Tensor = batch["layout"][:, :, 0]
        bbox_gt: Tensor = batch["layout"][:, :, 1]

        targets = {
            "layout": batch["layout"],
            "labels": label_gt,
            "boxes": bbox_gt,
        }

        if self.auxilary_task is None or self.auxilary_task == "uncond":
            pass

        else:

            if self.auxilary_task == "c":
                # Copy c
                random_layout[:, :, 0] = label_gt.clone()
                assert torch.all(random_layout[:, :, 0] == label_gt).item()
            elif self.auxilary_task == "cwh":
                # Copy c
                random_layout[:, :, 0] = label_gt.clone()
                assert torch.all(random_layout[:, :, 0] == label_gt).item()
                # Copy width and height -- order is {width, height, center_x, center_y}.
                random_layout[:, :, 1, 0:2] = bbox_gt[:, :, 0:2].clone()
                assert torch.all(
                    random_layout[:, :, 1, 0:2] == bbox_gt[:, :, 0:2]
                ).item()
            elif self.auxilary_task == "partial":
                # Keep first element, others are random layotus
                random_layout[:, 0, 0] = label_gt.clone()[:, 0]
                random_layout[:, 0, 1, 0:2] = bbox_gt.clone()[:, 0, 0:2]
            elif self.auxilary_task == "refinement":
                given_bbox = bbox_gt.clone()
                noise = torch.normal(0, 0.01, size=given_bbox.size()).type_as(
                    given_bbox
                )
                pad_mask = torch.sum(given_bbox, dim=-1) == 0.0
                noisy_bbox = torch.clamp(given_bbox + noise, min=0.0, max=1.0)
                noisy_bbox[pad_mask] = 0.0  # just in case
                random_layout = torch.stack([label_gt, noisy_bbox], dim=2)
            else:
                raise ValueError(f"Unknown auxilary task: {self.auxilary_task}")

            # Random shuffle
            N = random_layout.size(1)
            for i in range(random_layout.size(0)):
                indices = list(range(N))
                indices_shuffled = random.sample(indices, N)
                random_layout[i] = random_layout[i, indices_shuffled]

        new_inputs = {
            "image": batch["image_saliency"],
            "layout": random_layout,
        }

        return new_inputs, targets

    @abstractmethod
    def _encode_into_memory():
        raise NotImplementedError

    @abstractmethod
    def decode():
        raise NotImplementedError

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # layout: [bs, 32, 2, 4]
        img_feature, layout_feature = self._encode_into_memory(inputs)
        outputs = self.decode(img_feature, layout_feature)
        return outputs

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Tensor] = None,
        sampling_cfg: Optional[DictConfig] = None,
        return_violation: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        inputs, _ = self.preprocess(cond)
        # {pred_logits: [bs, max_elem, 4], pred_boxes: [bs, max_elem, 4]}
        outputs = self(inputs)
        output_seq = self.postprocess(outputs)

        if not return_violation:
            return output_seq
        return output_seq, None

    # keys = ["center_x", "center_y", "width", "height", "label", "mask"]を含めばOK
    def postprocess(self, outputs: dict) -> dict:
        if "bbox" in outputs:
            for i, key in enumerate(GEO_KEYS):
                outputs[key] = outputs["bbox"][..., i]
            del outputs["bbox"]
        else:
            for i, key in enumerate(GEO_KEYS):
                outputs[key] = outputs["pred_boxes"][..., i]
            del outputs["pred_boxes"]

            outputs["label"] = torch.argmax(outputs["pred_logits"], dim=-1)
            outputs["mask"] = outputs["label"] != (self.d_label - 1)
            del outputs["pred_logits"]

        return outputs

    def train_dis_loss(
        self, inputs: dict, targets: dict, outputs_gen: dict, discriminator
    ) -> tuple[None, dict]:
        # Discriminator loss using fake data
        logits_fake = discriminator(
            inputs["image"], outputs_gen["pred_layout"].detach()
        )
        loss_d_fake = (
            self.criterion_adv(logits_fake.view(-1), outputs_gen["fake_label"])
            * self.adv_weight
        )

        # Discriminator loss using real images
        logits_real = discriminator(inputs["image"], targets["layout"])
        loss_d_real = (
            self.criterion_adv(logits_real.view(-1), outputs_gen["real_label"])
            * self.adv_weight
        )

        losses = {
            "adv_fake": loss_d_fake,
            "adv_real": loss_d_real,
        }

        return (None, losses)

    def train_loss(
        self,
        inputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        discriminator: Optional[Any] = None,
        test: bool = False,
    ):
        losses = {}
        outputs = self(inputs)

        # reconstruction loss using bipartite graph matching
        targets_rec: list[dict[str, Tensor]] = [
            {"labels": c.long(), "boxes": b.float()}
            for c, b in zip(torch.argmax(targets["labels"], dim=-1), targets["boxes"])
        ]

        losses = self.criterion_rec(outputs, targets_rec)
        if not test and self.apply_weight:
            losses = {
                k: v * self.criterion_rec.weight_dict[k]
                for (k, v) in losses.items()
                if k != "loss_cardinality" and k in self.criterion_rec.weight_dict
            }

        # adversarial loss for real inputs
        if discriminator:
            bs = inputs["image"].size(0)
            all_real = torch.ones(bs, dtype=torch.float).type_as(outputs["pred_logits"])
            all_fake = torch.full((bs,), -1, dtype=torch.float).type_as(
                outputs["pred_logits"]
            )
            outputs["real_label"] = all_real
            outputs["fake_label"] = all_fake

            if outputs["pred_boxes"].size(-1) != targets["boxes"].size(-1):
                # Pad the prediction with zeros for bbox
                dim_pad = abs(
                    outputs["pred_boxes"].size(-1) - targets["boxes"].size(-1)
                )
                pad_box = torch.zeros_like(outputs["pred_boxes"])[..., :dim_pad]
                outputs["pred_boxes"] = torch.cat(
                    [outputs["pred_boxes"], pad_box], dim=-1
                )

            pred_layout = torch.concat(
                [
                    outputs["pred_logits"].unsqueeze(2),
                    outputs["pred_boxes"].unsqueeze(2),
                ],
                dim=2,
            )
            outputs["pred_layout"] = pred_layout

            logits_fake = discriminator(inputs["image"], pred_layout)
            loss_g_fake = self.criterion_adv(logits_fake.view(-1), all_real)
            losses["adv_fake"] = loss_g_fake * self.adv_weight

        return outputs, losses
