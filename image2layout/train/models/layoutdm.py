import logging
from typing import Any, Optional, Union

import datasets as ds
import torch
from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .common.base_model import BaseModel, ConditionalInputsForDiscreteLayout
from .common.image import ImageEncoder
from .common.retrieval_augment import RetrievalAugmentation
from .diffusion.discrete.constrained import ConstrainedMaskAndReplaceDiffusion
from .diffusion.discrete.default import MaskAndReplaceDiffusion
from .diffusion.discrete.logit_adjustment import set_weak_logits_for_refinement
from .diffusion.discrete.util import index_to_log_onehot, log_onehot_to_index

logger = logging.getLogger(__name__)


Q_TYPES = {
    "default": MaskAndReplaceDiffusion,
    "constrained": ConstrainedMaskAndReplaceDiffusion,
}
EMPTY_CONFIG = OmegaConf.create()


class LayoutDM(BaseModel):
    """
    Naively extending the discrete layout diffusion model to image-conditioned setting
    LayoutDM: Discrete Diffusion Model for Controllable Layout Generation [Inoue+, CVPR'23]
    https://arxiv.org/abs/2303.08137
    """

    def __init__(
        self,
        features: ds.Features,
        tokenizer: LayoutSequenceTokenizer,
        d_model: int = 256,
        num_timesteps: int = 50,
        pos_emb: str = "elem_attr",
        auxiliary_loss_weight: float = 1e-1,
        q_type: str = "constrained",
        retrieval_augmentation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        logger.info(f"{kwargs=}")

        self.features = features
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps
        self.retrieval_augmentation = retrieval_augmentation

        self.d_model = d_model
        num_layers = 6
        nhead = 8
        self.encoder = ImageEncoder(
            d_model=d_model,
            nhead=nhead,
            backbone_name="resnet50",
            num_layers=num_layers,
        )
        self.decoder = Q_TYPES[q_type](
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            tokenizer=tokenizer,
            num_timesteps=num_timesteps,
            pos_emb=pos_emb,
            auxiliary_loss_weight=auxiliary_loss_weight,
            **kwargs,
        )

        self.init_weight()

    def init_weight(self) -> None:
        logger.info("Initializing weights!")
        self.encoder.init_weight()
        self.decoder.init_weight()

    def forward(self, inputs: dict) -> dict[str, Tensor]:
        # since train / test discrepancy is too large, do not use this
        raise NotImplementedError

    def train_loss(
        self, inputs: dict, targets: dict, **kwargs: Any
    ) -> tuple[dict[str, Tensor]]:
        if self.retrieval_augmentation:
            memory = self._encode_into_memory(inputs["image"], inputs["retrieved"])
        else:
            memory = self._encode_into_memory(inputs["image"])
        # during training, loss is already computed inside self.decoder.forward
        return self.decoder(tgt=targets["seq"], memory=memory)  # type: ignore

    def _encode_into_memory(self, image: Tensor) -> Tensor:
        memory = self.encoder(image)
        return memory

    @torch.no_grad()
    def sample(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: DictConfig = EMPTY_CONFIG,
        cond_type: Optional[str] = "uncond",
        return_violation: bool = False,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        batch_size = batch_size or cond.image.size(0)

        d_label = self.tokenizer.N_total
        max_token_length = self.tokenizer.max_token_length

        if cond.task == "refinement":
            cond = set_weak_logits_for_refinement(cond, self.tokenizer, sampling_cfg)

        num_timesteps_eval = (
            sampling_cfg.num_timesteps
        )  # always set by validate_sampling_config
        assert num_timesteps_eval <= self.num_timesteps
        diffusion_list = []
        for i in range(num_timesteps_eval - 1, -1, -1):
            diffusion_list.append(int(i * self.num_timesteps / num_timesteps_eval))
        prev_diffusion_index = self.num_timesteps  # set very large value
        device = cond.image.device
        if cond.seq is not None:
            log_z = index_to_log_onehot(cond.seq, d_label)
        else:
            zero_logits = torch.zeros(
                (batch_size, d_label - 1, max_token_length), device=device
            )
            one_logits = torch.ones((batch_size, 1, max_token_length), device=device)
            mask_logits = torch.cat((zero_logits, one_logits), dim=1)
            log_z = torch.log(mask_logits)

        if self.retrieval_augmentation:
            memory = self._encode_into_memory(cond.image, cond.retrieved)
        else:
            memory = self._encode_into_memory(cond.image)

        for diffusion_index in diffusion_list:
            delta_t = prev_diffusion_index - diffusion_index
            assert delta_t > 0
            t = torch.full(
                (batch_size,), diffusion_index, device=device, dtype=torch.long
            )
            log_z = self.decoder.sample_single_step(
                log_z=log_z,
                memory=memory,
                model_t=t,
                skip_step=delta_t - 1,
                sampling_cfg=sampling_cfg,
                cond=cond,  # used to inject use-specified inputs
            )
            prev_diffusion_index = diffusion_index

        seq = log_onehot_to_index(log_z).cpu()
        output_seq = self.postprocess({"seq": seq})
        if not return_violation:
            return output_seq

        return output_seq, None

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:
        data = self.tokenizer.encode(inputs)
        image = torch.cat([inputs["image"], inputs["saliency"]], dim=1)
        inputs = {"image": image}  # seq and mask are generated on-the-fly later
        return inputs, {**data, "image": image}


class RetrievalAugmentedLayoutDM(LayoutDM):
    """
    Naively extending the discrete layout diffusion model to image-conditioned setting
    LayoutDM: Discrete Diffusion Model for Controllable Layout Generation [Inoue+, CVPR'23]
    https://arxiv.org/abs/2303.08137
    """

    def __init__(
        self,
        db_dataset,
        top_k: int,
        dataset_name: str,
        retrieval_backbone: str,
        random_retrieval: bool,
        saliency_k: Union[int, str],
        use_reference_image: bool,
        max_seq_length: int,
        **kwargs: Any,
    ) -> None:
        super(RetrievalAugmentedLayoutDM, self).__init__(
            retrieval_augmentation=True, **kwargs
        )

        self.top_k = top_k
        self.random_retrieval = random_retrieval

        self.retrieval_aug = RetrievalAugmentation(
            d_model=self.d_model,
            top_k=top_k,
            dataset_name=dataset_name,
            num_classes=self.features["label"].feature.num_classes,
            max_seq_length=max_seq_length,
            use_reference_image=use_reference_image,
        )

    def train_loss(
        self, inputs: dict, targets: dict, **kwargs: Any
    ) -> tuple[dict[str, Tensor]]:
        memory = self._encode_into_memory(
            inputs["image"],
            inputs["retrieved"],
        )
        # during training, loss is already computed inside self.decoder.forward
        return self.decoder(tgt=targets["seq"], memory=memory)  # type: ignore

    def _encode_into_memory(
        self, image: Tensor, retrieved: dict[str, Tensor]
    ) -> Tensor:
        memory = super()._encode_into_memory(image)

        # Retrieval Augmentation
        memory = self.retrieval_aug(
            image_backbone=self.encoder,
            img_feature=memory,
            retrieved_layouts=retrieved,
        )  # [b, seq, c]

        return memory

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:

        new_inputs, targets = super().preprocess(inputs)

        retrieved_samples = self.retrieval_aug.preprocess_retrieved_samples(
            inputs["retrieved"]
        )
        new_inputs["retrieved"] = retrieved_samples

        return new_inputs, targets
