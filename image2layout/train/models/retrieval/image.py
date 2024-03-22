import logging
import os

import datasets as ds
import fsspec
import numpy as np
import timm
import torch
import torch.nn.functional as F
from dreamsim import dreamsim
from einops import rearrange
from image2layout.train.helpers.rich_utils import get_progress
from torch import Tensor
from torchvision import transforms

logger = logging.getLogger(__name__)

KEYS = [
    "image",
    "saliency",
    "center_x",
    "center_y",
    "width",
    "height",
    "label",
    "mask",
]

DEEP_BACKBONES = {
    "clip": "hf_hub:timm/vit_base_patch16_clip_224.openai",
    "vgg": "hf_hub:timm/vgg16.tv_in1k",
}


def coarse_saliency(saliency: Tensor, size: tuple[int, int] = (16, 16)) -> np.ndarray:
    # note: resolution is specific to current setup
    assert saliency.size(1) == 350 and saliency.size(2) == 240

    h = rearrange(saliency, "1 h w -> 1 1 h w")
    h = F.interpolate(h, size=size)
    h = h.flatten()
    h = torch.clamp(h, 0.0, 1.0)
    h = 2 * h - 1.0  # in [-1.0, 1.0] range for applying any similarity metric
    return h.numpy()  # type: ignore


class FeatureExtracterBackbone:
    """An abstract model that defines general interface for image-to-layout models"""

    def __init__(self, db_dataset: ds.Dataset, retrieval_backbone: str) -> None:
        self.db_dataset = db_dataset
        self.retrieval_backbone = retrieval_backbone
        logger.info(f"Build RetrievalBackbone with {retrieval_backbone=}")

        if self.retrieval_backbone in DEEP_BACKBONES.keys():
            backbone_tag = DEEP_BACKBONES[retrieval_backbone]
            self.model = timm.create_model(backbone_tag, pretrained=True, num_classes=0)

            data_config = timm.data.resolve_model_data_config(self.model)
            transform = timm.data.create_transform(**data_config, is_training=False)
            transform = [
                t
                for t in transform.transforms
                if not isinstance(t, transforms.ToTensor)
            ]
            self.transform = transforms.Compose(transform)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            self.model.cuda()

        elif self.retrieval_backbone == "dreamsim":

            cache_dir = f"{os.path.expanduser('~')}/.cache/image2layout/dreamsim"
            fs, path_prefix = fsspec.core.url_to_fs(cache_dir)
            if not fs.exists(path_prefix):
                fs.makedirs(path_prefix)

            self.model, _ = dreamsim(pretrained=True, cache_dir=cache_dir)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            self.model.cuda()

            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                ]
            )

            self.model = self.model.embed
        elif self.retrieval_backbone == "saliency":
            pass
        else:
            raise ValueError(f"{retrieval_backbone=} is not supported")

    @torch.no_grad()
    def image_to_feature(self, img: Tensor) -> np.ndarray:
        if len(img.size()) == 3:
            img = img.unsqueeze(0)
        assert img.size(1) == 3, img.size()
        image = self.transform(img).cuda()
        features = self.model(image).cpu().numpy()  # [1, 512]
        return features[0]

    def extract_dataset_features(self) -> np.ndarray:
        _vectors = []
        pbar = get_progress(
            self.db_dataset,
            f"[{self.retrieval_backbone}] extract features",
            True,
        )
        for example in pbar:
            if self.retrieval_backbone != "saliency":
                feat = self.image_to_feature(example["image"])
            else:
                feat = coarse_saliency(example["saliency"].cpu())
            _vectors.append(feat)
        vectors = np.array(_vectors)
        return vectors

    @torch.no_grad()
    def get_query(self, batch: dict[str, Tensor]) -> np.ndarray:
        if self.retrieval_backbone == "saliency":
            query: np.ndarray = coarse_saliency(batch["saliency"].cpu())
        else:
            query: np.ndarray = self.image_to_feature(batch["image"])
        return query
