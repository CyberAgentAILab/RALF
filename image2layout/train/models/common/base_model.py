import logging
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

import datasets as ds
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import BoolTensor, LongTensor, Tensor

logger = logging.getLogger(__name__)


@dataclass
class _BaseConditionalInputs:
    """
    All-in-one container for conditional inputs in sampling.
    """

    image: Tensor  # (B, 4, H, W), concat. of image and saliency
    id: Optional[LongTensor]  # (B,), unique id for each sample
    task: Optional[str] = None  # see COND_TYPES in task.py

    def tensor_like_attributes(self) -> Iterable[tuple[str, Tensor]]:
        """
        Utility to iterate over tensor-like attributes.
        """
        for name in dir(self):
            if name.startswith("__"):
                continue
            attr = getattr(self, name)
            if not callable(attr) and torch.is_tensor(attr):  # type: ignore
                yield (name, attr)

    def to(self, x: Any) -> "_BaseConditionalInputs":
        """
        Tensor dtype and/or device conversion at once for all tensor-like attributes.
        """
        for name, attr in self.tensor_like_attributes():
            setattr(self, name, attr.to(x))
        return self

    def duplicate(self, n: int) -> "_BaseConditionalInputs":
        """
        Duplicate the batch dimension for n times.
        This is usually used to sample multiple outputs per sample.
        """
        for name, attr in self.tensor_like_attributes():
            setattr(self, name, repeat(attr, "b ... -> (b n) ...", n=n))
        return self


@dataclass
class ConditionalInputsForDiscreteLayout(_BaseConditionalInputs):
    """
    Geometry-related layout information is provided as discrete tokens
    If mask[i, j] is True, it is provided by a user and should be used.
    """

    seq: Optional[LongTensor] = None  # (B, max_token_length)
    mask: Optional[BoolTensor] = None  # (B, max_token_length)

    # used only for refinement
    seq_observed: Optional[LongTensor] = None  # (B, max_token_length)
    weak_mask: Optional[BoolTensor] = None  # (B, max_token_length)
    weak_logits: Optional[Tensor] = None  # (B, max_token_length)

    # used only for relation (dense data format for simplicity)
    # E = max_seq_length * (max_seq_length + 1) // 2
    edge_indexes: Optional[LongTensor] = None  # (B, E, 2)
    edge_attributes: Optional[LongTensor] = None  # (B, E)


@dataclass
class RetrievalAugmentedConditionalInputsForDiscreteLayout(
    ConditionalInputsForDiscreteLayout
):
    # used for retrieval-augmented models
    # K: number of top-k retrieved samples
    retrieved: dict[str, Tensor] = field(
        default_factory=lambda: {}
    )  # each field has shape of (B, K, ...)

    def __post_init__(self) -> None:
        # TODO: check if this is really working
        # TODO: Why does ImageEncoder assume that image is concatenated with saliency?

        if self.retrieved["image"].size(2) >= 4:
            return

        self.retrieved["image"] = torch.cat(
            [
                self.retrieved["image"],
                self.retrieved["saliency"],
            ],
            dim=2,
        )  # type: ignore

    def to(self, x: Any) -> "RetrievalAugmentedConditionalInputsForDiscreteLayout":
        for name, attr in self.tensor_like_attributes():
            setattr(self, name, attr.to(x))
        # self.retrieved is a dict of tensors
        for k, v in self.retrieved.items():
            if isinstance(v, Tensor):
                self.retrieved[k] = v.to(x)
        return self


@dataclass
class ConditionalInputsForContinuousLayout(_BaseConditionalInputs):
    # seq: Optional[LongTensor] = None  # (B, max_token_length)  # to be implmented
    mask: Optional[BoolTensor] = None  # (B, max_token_length)


class BaseModel(nn.Module, metaclass=ABCMeta):
    """An abstract model that defines general interface for image-to-layout models"""

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def features(self) -> ds.Features:
        return self._features

    @features.setter
    def features(self, value: Any) -> None:
        self._features = value

    @property
    def special_token_ids(self):
        sp_tokens = self.tokenizer.special_tokens
        ids = {key: self.tokenizer.name_to_id(key) for key in sp_tokens}
        if "mask" not in ids:
            ids["mask"] = -1
        return ids

    def forward(self, inputs: dict) -> dict[str, Tensor]:
        raise NotImplementedError

    def aggregate_sampling_config(
        self, sampling_cfg: DictConfig, test_cfg: Optional[DictConfig] = None
    ) -> DictConfig:
        """
        Aggregate sampling config from test config.
        """
        OmegaConf.set_struct(sampling_cfg, True)

        model_type = type(self).__name__
        is_logit_adjustable = bool(re.search(r"LayoutDM|MaskGIT|Autoreg", model_type))
        is_diffusion = bool(re.search(r"LayoutDM", model_type))
        is_non_autoregressive = bool(re.search(r"LayoutDM|MaskGIT", model_type))

        if is_non_autoregressive and "num_timesteps" not in sampling_cfg:
            if (tokenizer := getattr(self, "tokenizer", None)) is not None:
                with open_dict(sampling_cfg):  # type: ignore
                    sampling_cfg.num_timesteps = tokenizer.max_token_length

        if test_cfg is not None:
            if is_logit_adjustable:
                # solve refinement using logit adjustment (autoreg and layoutdm)
                if test_cfg.cond_type == "refinement":
                    assert test_cfg.refine_lambda > 0.0
                    for name in ["mode", "offset_ratio", "lambda"]:
                        key = f"refine_{name}"
                        with open_dict(sampling_cfg):
                            sampling_cfg[key] = test_cfg[key]

            if is_diffusion:
                if test_cfg.cond_type == "relation":
                    assert test_cfg.relation_lambda > 0.0
                    # solve relation using logit adjustment (layoutdm only)
                    for name in ["mode", "lambda", "tau", "num_update"]:
                        key = f"relation_{name}"
                        with open_dict(sampling_cfg):
                            sampling_cfg[key] = test_cfg[key]

        return sampling_cfg

    @torch.no_grad()  # type: ignore
    def sample(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = 1,
        sampling_cfg: Optional[DictConfig] = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        raise NotImplementedError

    def generator_loss(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:
        """
        Do model-specific preprocessing (e.g., bbox tokenization) and return dict
        for input and target
        """
        raise NotImplementedError

    def optim_groups(
        self,
        base_lr: Optional[float] = None,
        weight_decay: float = 0.0,
        forced_no_weight_decay: Optional[list[str]] = None,
        custom_lr: Optional[dict[str, float]] = None,
    ) -> Iterable[dict]:
        # see https://github.com/kampta/DeepLayout/blob/main/layout_transformer/model.py#L139
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.modules.activation.MultiheadAttention,
            nn.Conv2d,
            nn.Conv1d,
            nn.Parameter,
        )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                if _p.requires_grad is False:
                    continue
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                # LSTM parameterss
                elif "weight_ih" in pn or "weight_hh" in pn:
                    decay.add(fpn)
                elif "bias_ih" in pn or "bias_hh" in pn:
                    no_decay.add(fpn)

        if forced_no_weight_decay:
            for k in forced_no_weight_decay:
                no_decay.add(k)

        # validate that we considered every parameter
        # for pn, p in self.named_parameters():
        #     if p.requires_grad is False:
        #         print(pn, p.requires_grad)
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad is True
        }  # TODO: ok?
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        if custom_lr and len(custom_lr) > 0:
            decay_custom, no_decay_custom = set(), set()

            optim_groups = []
            for prefix, lr in custom_lr.items():
                logger.info(f"Custom learning rate for {prefix=} {lr=}")

                logger.info("Add parameters with weight decay")
                params = []
                for pn in sorted(list(decay)):
                    if pn.startswith(prefix):
                        decay_custom.add(pn)
                        params.append(param_dict[pn])
                if len(params) > 0:
                    optim_groups.append(
                        {
                            "params": params,
                            "weight_decay": weight_decay,
                            "lr": lr,
                        }
                    )

                logger.info("Add parameters without weight decay")
                params = []
                for pn in sorted(list(no_decay)):
                    if pn.startswith(prefix):
                        no_decay_custom.add(pn)
                        params.append(param_dict[pn])
                if len(params) > 0:
                    optim_groups.append(
                        {
                            "params": params,
                            "weight_decay": 0.0,
                            "lr": lr,
                        }
                    )

            # # for debugging
            # logger.info(f"{decay_custom=}")
            # logger.info(f"{no_decay_custom=}")

            decay_default = decay - decay_custom
            if len(decay_default) > 0:
                optim_groups.append(
                    {
                        "params": [
                            param_dict[pn] for pn in sorted(list(decay_default))
                        ],
                        "weight_decay": weight_decay,
                        "lr": base_lr,
                    }
                )

            no_decay_default = no_decay - no_decay_custom
            if len(no_decay_default) > 0:
                optim_groups.append(
                    {
                        "params": [
                            param_dict[pn] for pn in sorted(list(no_decay_default))
                        ],
                        "weight_decay": 0.0,
                        "lr": base_lr,
                    }
                )
        else:
            # create the pytorch optimizer object
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": weight_decay,
                    "lr": base_lr,
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                    "weight_decay": 0.0,
                    "lr": base_lr,
                },
            ]

        return optim_groups

    def update_per_epoch(
        self, epoch: int, freeze_dis_epoch: int, max_epoch: int
    ) -> None:
        """
        Update some non-trainable parameters during training (e.g., warmup)
        """
        pass

    def compute_params(self, layer, name="") -> None:
        n_params = sum([p.numel() for p in layer.parameters()]) / 1e6
        logger.info(
            f"[{name}] [{type(layer).__name__}] number of parameters: {n_params:.2f}M"
        )

    def compute_stats(self) -> None:
        n_params = sum([p.numel() for p in self.parameters()]) / 1e6
        logger.info(f"number of parameters: {n_params:.2f}M")

    def postprocess(self, outputs: dict) -> dict:
        """
        Do model-specific postprocessing (e.g., bbox tokenization)
        and return dict for each attribute (batched)
        """
        if (tokenizer := self.tokenizer) is not None:
            if "seq" in outputs:
                seq = outputs["seq"]  # (B, S)
            else:
                logits = outputs["logits"]  # (B, S, C)
                invalid = repeat(
                    ~tokenizer.token_mask, "x c -> b x c", b=logits.size(0)  # type: ignore
                )

                # note: diffusion models logits assume (B, C, S)
                if logits.size(-1) == tokenizer.max_token_length:  # type: ignore
                    logits = rearrange(logits, "b c s -> b s c")
                logits[invalid] = -float("Inf")
                seq = torch.argmax(logits, dim=-1)
            outputs = tokenizer.decode(seq)  # type: ignore
        else:
            raise NotImplementedError
        return outputs


def main() -> None:
    # check positional encoding
    import matplotlib.pyplot as plt
    from image2layout.train.models.common.positional_encoding import (
        build_position_encoding_1d,
    )

    pos_emb = build_position_encoding_1d("layout", d_model=256).eval()
    # pos_emb = build_position_encoding_1d(
    #     "elem_attr", d_model=256, n_attr_per_elem=5
    # ).eval()
    x = torch.zeros(1, 50, 256)  # set all zero will just output positional encodings
    with torch.no_grad():
        y = pos_emb(x)
    cax = plt.matshow(y[0].detach().numpy())
    plt.gcf().colorbar(cax)
    plt.savefig("dummy.pdf")


if __name__ == "__main__":
    main()
