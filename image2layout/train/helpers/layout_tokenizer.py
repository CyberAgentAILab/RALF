import logging
import os
import pickle
from copy import deepcopy
from typing import Any, Optional

import datasets as ds
import fsspec
import torch
from einops import rearrange, reduce, repeat
from image2layout.train.global_variables import GEO_KEYS, PRECOMPUTED_WEIGHT_DIR
from image2layout.train.helpers.bucketizer import (
    BaseBucketizer,
    bucketizer_factory,
    get_kmeans_cluster_center,
)
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from torch import BoolTensor, Tensor

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_VOCABULARIES = ["pad", "bos", "eos", "mask"]
CHOICES = {
    "var_order": [
        ["label", "width", "height", "center_x", "center_y"],
        ["label", "center_x", "center_y", "width", "height"],
    ],
    "geo_quantization": [
        "linear",
        "kmeans",
    ],
    "is_loc_vocab_shared": [True, False],
}

_TORCH_PADDING_VALUE_FACTORY = {
    torch.int64: 0,
    torch.float32: 0.0,
    torch.bool: False,
}


def init_layout_tokenizer(
    tokenizer_cfg: DictConfig,
    dataset_cfg: DictConfig,
    label_feature: ds.ClassLabel,
) -> "LayoutSequenceTokenizer":
    max_seq_length = dataset_cfg.max_seq_length
    if tokenizer_cfg.geo_quantization == "kmeans":
        if "pku" in dataset_cfg.name:
            _name = (
                f"cache/{dataset_cfg.name}{max_seq_length}_kmeans_train_clusters.pkl"
            )
        else:
            _name = f"cache/{dataset_cfg.name}_kmeans_train_clusters.pkl"
        if fsspec.filesystem("file").exists(_name):
            weight_path = _name
        else:
            weight_path = os.path.join(PRECOMPUTED_WEIGHT_DIR, "clustering", _name)
    else:
        weight_path = None

    return LayoutSequenceTokenizer(
        label_feature=label_feature,
        max_seq_length=max_seq_length,
        weight_path=weight_path,
        **OmegaConf.to_container(tokenizer_cfg),
    )


def padding_value_factory(dtype: Any) -> Any:
    return _TORCH_PADDING_VALUE_FACTORY[dtype]


def _pad_sequence(seq: Tensor, max_seq_length: int) -> Tensor:
    dim = -1
    new_shape = list(seq.shape)
    s = max_seq_length - new_shape[dim]
    if s > 0:
        new_shape[dim] = s
        dtype = seq.dtype
        value = padding_value_factory(dtype)
        pad = torch.full(new_shape, value, dtype=dtype)
        new_seq = torch.cat([seq, pad], dim=dim)
    else:
        new_seq = seq

    return new_seq


class LayoutTokenizer:
    """
    Tokenizer converts inputs into (dict of) a sequence
    This is a base class for all tokenizers
    """

    def __init__(
        self,
        label_feature: ds.ClassLabel,
        max_seq_length: int,
        # below are similar with TokenizerConfig
        num_bin: int = 32,
        var_order: list[str] = (
            "label",
            "width",
            "height",
            "center_x",
            "center_y",
        ),  # type: ignore
        pad_until_max: bool = (
            False  # True for diffusion models, False for others for efficient batching
        ),
        special_tokens: list[str] = ("pad", "bos", "eos"),  # type: ignore
        is_loc_vocab_shared: bool = False,
        geo_quantization: str = "linear",
        weight_path: Optional[str] = None,
        **kwargs: Any,  # to ignore previously used arguments
    ) -> None:
        # note: these values are not overridden. To get values, please use getter.
        self._label_feature = label_feature
        self._max_seq_length = max_seq_length
        self._num_bin = num_bin
        self._var_order = var_order
        self._pad_until_max = pad_until_max
        self._special_tokens = special_tokens
        self._is_loc_vocab_shared = is_loc_vocab_shared
        self._geo_quantization = geo_quantization

        logger.info(f"Initialize {self.__class__.__name__} with {num_bin=}")

        # validation
        for (key, seq) in CHOICES.items():
            assert getattr(self, f"_{key}") in seq  # type: ignore

        assert "pad" in self.special_tokens
        assert all(token in SPECIAL_TOKEN_VOCABULARIES for token in self.special_tokens)
        if "mask" in self.special_tokens:
            assert self.special_tokens.index("mask") == self.N_sp_token - 1

        if self.geo_quantization == "kmeans":
            assert weight_path

        self._bucketizers = {}
        if self.geo_quantization == "kmeans":
            fs, path_prefix = fsspec.core.url_to_fs(weight_path)
            logger.info(f"Load {weight_path=}")
            with fs.open(path_prefix, "rb") as f:
                weights: dict[str, KMeans] = pickle.load(f)

        for key in self.var_order:
            if key == "label":
                continue

            bucketizer = bucketizer_factory(self.geo_quantization)
            bucketizer_args = {"n_boundaries": self._num_bin}
            if self.geo_quantization == "kmeans":
                bucketizer_args["cluster_centers"] = get_kmeans_cluster_center(
                    key=f"{key}-{self._num_bin}", weights=weights
                )  # type: ignore
            self._bucketizers[key] = bucketizer(**bucketizer_args)

        detail = f"{self.N_label},{self.N_bbox},{self.N_sp_token}"
        logger.info(f"N_total={self.N_total},(N_label, N_bbox, N_sp_token)=({detail})")
        self._special_token_name_to_id = {
            token: self.special_tokens.index(token) + self.N_label + self.N_bbox
            for token in self.special_tokens
        }
        self._special_token_id_to_name = {
            v: k for (k, v) in self._special_token_name_to_id.items()
        }

    def _fill_until_max_seq_length(
        self, inputs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Some models such as diffusion models require fixed-length inputs.
        To do so, just add some padding tokens at the end of each sequence.
        """
        if self._pad_until_max:
            for (key, value) in inputs.items():
                inputs[key] = _pad_sequence(value, self.max_seq_length)
        return inputs

    def _insert_padding_token(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        torch's default collate function may pad a sequence with 0 / 0.0 / False.
        To avoid unintended effects, we replace the padded values with the id of 'pad',
        according to "mask" attributes that are always correct (by manual procedure).
        """
        if "pad" in self.special_tokens:
            pad_mask = ~inputs["mask"]
            pad_id = self.name_to_id("pad")

            inputs["label"][pad_mask] = pad_id
            for key in GEO_KEYS:
                inputs[key][pad_mask] = pad_id

        return inputs

    def _detect_oov(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        If a set of tokens for an element is corrupted (out-of-vocabulary), discard the element.
        it returns a boolean element-level mask.
        """
        label_valid = (0 <= inputs["label"]) & (inputs["label"] < self.N_label)

        geo_valid = torch.full(label_valid.size(), fill_value=True)
        for key in GEO_KEYS:
            valid = (0 <= inputs[key]) & (inputs[key] < self.N_bbox)
            geo_valid &= valid

        invalid = torch.logical_not(label_valid & geo_valid)
        return invalid

    def _detect_eos(self, label: Tensor) -> Tensor:
        """
        If a tokenizer use BOS/EOS,
        it returns a boolean element-level mask indicating a position of EOS.
        """
        if "bos" in self.special_tokens and "eos" in self.special_tokens:
            invalid = torch.cumsum(label == self.name_to_id("eos"), dim=1) > 0
        else:
            invalid = torch.full(label.size(), fill_value=False)
        return invalid

    # functions below are for accesing special token properties
    def name_to_id(self, name: str) -> int:
        return self._special_token_name_to_id[name]

    def id_to_name(self, id_: int) -> str:
        return self._special_token_id_to_name[id_]

    @property
    def bucketizers(self) -> dict[str, BaseBucketizer]:
        return self._bucketizers

    @property
    def geo_quantization(self) -> str:
        return self._geo_quantization

    @property
    def is_loc_vocab_shared(self) -> bool:
        return self._is_loc_vocab_shared

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @property
    def max_token_length(self) -> int:
        return self.max_seq_length * self.N_var_per_element

    @property
    def N_bbox(self) -> int:
        if self.is_loc_vocab_shared:
            return self.N_bbox_per_var
        else:
            return self.N_bbox_per_var * 4

    @property
    def N_bbox_per_var(self) -> int:
        return self._num_bin

    @property
    def N_label(self) -> int:
        return int(self._label_feature.num_classes)

    @property
    def N_sp_token(self) -> int:
        return len(self.special_tokens)

    @property
    def N_total(self) -> int:
        return self.N_label + self.N_bbox + self.N_sp_token

    @property
    def N_var_per_element(self) -> int:
        return len(self.var_order)

    @property
    def pad_until_max(self) -> bool:
        return self._pad_until_max

    @property
    def special_tokens(self) -> list[str]:
        return self._special_tokens

    @property
    def var_order(self) -> list[str]:
        return self._var_order


class LayoutSequenceTokenizer(LayoutTokenizer):
    """
    Converts a layout into a sequence (c_1, x_1, y_1, w_1, h_1, c_2, ...)
    Please refer to LayoutTokenizer for keyword arguments.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def encode(
        self,
        inputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Parameters:
            inputs: dict of the following items
                label (LongTensor): shape (B, S)
                center_x (FloatTensor): shape (B, S)
                center_y (FloatTensor): shape (B, S)
                width (FloatTensor): shape (B, S)
                height (FloatTensor): shape (B, S)
                mask (BoolTensor): shape (B, S)
        Returns:
            outputs: dict of the following items
                seq (LongTensor): shape (B, 5 * S)
                mask (BoolTensor): shape (B, 5 * S)
        """
        data = {}
        data["label"] = deepcopy(inputs["label"])
        for i, key in enumerate(GEO_KEYS):
            data[key] = self._bucketizers[key].encode(inputs[key])
            data[key] += self.N_label
            if not self._is_loc_vocab_shared:
                data[key] += i * self.N_bbox_per_var

        data["mask"] = deepcopy(inputs["mask"])

        data = self._fill_until_max_seq_length(data)
        data = self._insert_padding_token(data)

        B, S = data["label"].size()[:2]
        C = self.N_var_per_element

        # sanity check
        seq_len = reduce(data["mask"].int(), "b s -> b 1", reduction="sum")
        indices = rearrange(torch.arange(0, S), "s -> 1 s")

        assert torch.all(torch.logical_not(data["mask"]) == (seq_len <= indices)).item()

        # make 1d sequence
        seq = torch.stack([data[key] for key in self.var_order], dim=-1)
        seq = rearrange(seq, "b s x -> b (s x)")
        mask = repeat(data["mask"], "b s -> b (s c)", c=C).clone()

        # append BOS/EOS for autoregressive models
        if "bos" in self.special_tokens and "eos" in self.special_tokens:
            indices = rearrange(torch.arange(0, S * C), "s -> 1 s")
            eos_mask = seq_len * C == indices
            seq[eos_mask] = self.name_to_id("eos")
            mask[eos_mask] = True

            # add [BOS] at the beginning (length is incremented)
            bos = torch.full((B, 1), self.name_to_id("bos"))
            seq = torch.cat([bos, seq], axis=-1)  # type: ignore
            mask = torch.cat([torch.full((B, 1), fill_value=True), mask], axis=-1)  # type: ignore

        outputs = {"seq": seq, "mask": mask}
        return outputs

    def decode(self, seq: Tensor) -> dict[str, Tensor]:
        """
        Parameters:
            seq (LongTensor): (B, 5 * S)

        Returns:
            outputs: dict of the following items
                label: torch.LongTensor of shape (B, S)
                center_x: torch.FloatTensor of shape (B, S)
                center_y: torch.FloatTensor of shape (B, S)
                width: torch.FloatTensor of shape (B, S)
                height: torch.FloatTensor of shape (B, S)
                mask: torch.BoolTensor of shape (B, S)
        """
        seq = rearrange(
            deepcopy(seq),
            "b (s c) -> b s c",
            c=self.N_var_per_element,
        )
        outputs = {}
        for i, key in enumerate(self.var_order):
            outputs[key] = seq[..., i]
            if key in GEO_KEYS:
                outputs[key] = outputs[key] - self.N_label
                if not self._is_loc_vocab_shared:
                    mult = GEO_KEYS.index(key)
                    outputs[key] = outputs[key] - mult * self.N_bbox_per_var

        invalid = self._detect_eos(outputs["label"])
        invalid = invalid | self._detect_oov(outputs)

        for key in GEO_KEYS:
            outputs[key][invalid] = 0
            outputs[key] = self._bucketizers[key].decode(outputs[key])

        for key in self.var_order:
            padding_value = padding_value_factory(outputs[key].dtype)
            outputs[key][invalid] = padding_value

        outputs["mask"] = torch.logical_not(invalid)
        return outputs

    @property
    def token_mask(self) -> Tensor:
        """
        Returns a bool tensor in shape (S, C), which is used to filter our invalid predictions
        E.g., predict high probs on x=1, while the loc. token is for predicting a category
        """
        ng_tokens = ["bos", "mask"]  # shouldn't be predicted
        last = BoolTensor(
            [False if x in ng_tokens else True for x in self.special_tokens]
        )

        # get masks for geometry variables
        masks = {}
        if self.is_loc_vocab_shared:
            for key in GEO_KEYS:
                masks[key] = torch.full((self.N_bbox_per_var,), True)
        else:
            false_tensor = torch.full((self.N_bbox,), False)
            for key in self.var_order:
                if key == "label":
                    continue
                tensor = deepcopy(false_tensor)
                mult = GEO_KEYS.index(key)
                start, stop = (
                    mult * self.N_bbox_per_var,
                    (mult + 1) * self.N_bbox_per_var,
                )
                tensor[start:stop] = True
                masks[key] = torch.cat(
                    [torch.full((self.N_label,), False), tensor, last]
                )

        masks["label"] = torch.cat(
            [
                torch.full((self.N_label,), True),
                torch.full((self.N_bbox,), False),
                last,
            ]
        )

        mask = torch.stack([masks[k] for k in self.var_order], dim=0)
        mask = repeat(mask, "x c -> (s x) c", s=self.max_seq_length)
        return mask
