import math

import torch
from einops import repeat
from image2layout.train.helpers.layout_tokenizer import (
    GEO_KEYS,
    LayoutSequenceTokenizer,
)
from torch import LongTensor, Tensor


def _bucketize(
    inputs: LongTensor,
    from_ids: LongTensor,
    to_ids: LongTensor,
) -> LongTensor:
    """
    Map a set of ids to a different set of ids
    """
    assert from_ids.size() == to_ids.size()
    assert set(inputs.unique().tolist()) <= set(from_ids.tolist())  # type: ignore
    index = torch.bucketize(inputs.ravel(), from_ids)
    remapped = to_ids[index].reshape(inputs.shape).to(inputs)
    return remapped  # type: ignore


class Converter:
    """
    In constrained diffusion, we should handle both transition matrix:
    - full: for all tokens
    - partial: for each attribute
    Note: f and p is short for full and partial, respectively.
    """

    def __init__(
        self,
        tokenizer: LayoutSequenceTokenizer,
    ):
        assert tokenizer.special_tokens == ["pad", "mask"]
        # note: this is unnecesary in principle but for ease of implementation
        assert tokenizer.var_order[0] == "label"
        C = tokenizer.N_var_per_element

        pad_id = tokenizer.name_to_id("pad")
        mask_id = tokenizer.name_to_id("mask")

        mapping = {}
        mapping["label"] = {
            "partial": list(range(tokenizer.N_label + 2)),
            "full": list(range(tokenizer.N_label)) + [pad_id, mask_id],
        }

        # for at_once mathods
        offset = {}
        offset["normal_f_to_p"] = [
            0,
        ]
        offset["special_f_to_p"] = [-tokenizer.N_bbox] + [
            -pad_id + tokenizer.N_bbox_per_var for _ in range(C - 1)
        ]
        offset["boundary_f_to_p"] = [pad_id for _ in range(C)]
        offset["normal_p_to_f"] = [
            0,
        ]
        offset["special_p_to_f"] = [
            tokenizer.N_bbox,
        ] + [pad_id - tokenizer.N_bbox_per_var for _ in range(C - 1)]
        offset["boundary_p_to_f"] = [
            tokenizer.N_label,
        ] + [tokenizer.N_bbox_per_var for _ in range(C - 1)]

        for key in tokenizer.var_order:
            if key == "label":
                continue
            num_bin = tokenizer.N_bbox_per_var
            special_ids = [tokenizer.name_to_id("pad"), tokenizer.name_to_id("mask")]

            start = tokenizer.N_label
            if not tokenizer.is_loc_vocab_shared:
                i = GEO_KEYS.index(key)
                start += i * num_bin
            mapping[key] = {
                "partial": list(range(num_bin + 2)),
                "full": list(range(start, start + num_bin)) + special_ids,
            }
            offset["normal_f_to_p"].append(-start)
            offset["normal_p_to_f"].append(start)

        # self._mapping: dict[str, LongTensor] = {}
        # for k, v in mapping.items():
        #     self._mapping[k] = {x: LongTensor(y) for (x, y) in v.items()}
        self._mapping = {
            k: {x: LongTensor(y) for (x, y) in v.items()} for (k, v) in mapping.items()  # type: ignore
        }
        # for k, v in _offset.items():
        #     _offset[k] = LongTensor(v)
        offset = {k: LongTensor(v) for (k, v) in offset.items()}  # type: ignore

        # pre-allocate to avoid calling repeat in every call
        self._batched_mapping, self._batched_offset = {}, {}
        B, S = 512, tokenizer.max_seq_length
        self._batched_mapping = {
            k: {x: repeat(y, "c -> b c s", b=B, s=S) for (x, y) in v.items()}
            for (k, v) in self._mapping.items()
        }
        self._batched_offset = {
            k: repeat(v, "x -> b s x", b=B, s=tokenizer.max_seq_length)
            for (k, v) in offset.items()
        }

        self._tokenizer = tokenizer
        self._C = C

    def __call__(self) -> None:
        raise NotImplementedError

    def p_to_f_id(self, inputs: LongTensor, key: str) -> LongTensor:
        outputs = _bucketize(
            inputs=inputs,
            from_ids=self._mapping[key]["partial"],
            to_ids=self._mapping[key]["full"],
        )
        return outputs

    def p_to_f_id_all(self, ids_p: LongTensor) -> LongTensor:
        """
        p_to_f_id for all the layout tokens at once for efficiency.
        Note: the shape of ids is (B, S, C), where C = len(tokenizer.N_var_per_element),
        """
        B, S, C = ids_p.size()
        assert C == self._C
        ids_normal_f = ids_p + self._batched_offset["normal_p_to_f"][:B, :S]
        ids_special_f = ids_p + self._batched_offset["special_p_to_f"][:B, :S]
        ids_f: LongTensor = torch.where(
            ids_p < self._batched_offset["boundary_p_to_f"][:B, :S],
            ids_normal_f,
            ids_special_f,
        )  # type: ignore
        return ids_f

    def f_to_p_id(self, inputs: LongTensor, key: str) -> LongTensor:
        outputs = _bucketize(
            inputs=inputs,
            from_ids=self._mapping[key]["full"],
            to_ids=self._mapping[key]["partial"],
        )
        return outputs

    def f_to_p_id_all(self, ids_f: LongTensor) -> LongTensor:
        """
        f_to_p_id for all the layout tokens at once for efficiency.
        Note: the shape of ids is (B, S, C), where C = len(tokenizer.N_var_per_element),
        """
        B, S, C = ids_f.size()
        assert C == self._C
        assert B <= 512
        ids_normal_p = ids_f + self._batched_offset["normal_f_to_p"][:B, :S]
        ids_special_p = ids_f + self._batched_offset["special_f_to_p"][:B, :S]
        ids_p: LongTensor = torch.where(
            ids_f < self._batched_offset["boundary_f_to_p"][:B, :S],
            ids_normal_p,
            ids_special_p,
        )  # type: ignore
        return ids_p

    def p_to_f_log(self, inputs: Tensor, key: str) -> Tensor:
        B, _, S = inputs.size()
        assert B <= 512
        shape = (B, self._tokenizer.N_total, S)
        outputs = torch.full(shape, math.log(1e-30)).to(inputs)
        index = self._batched_mapping[key]["full"][:B, :, :S]
        outputs.scatter_(dim=1, index=index, src=inputs)
        return outputs

    def f_to_p_log(
        self,
        inputs: Tensor,
        key: str,
    ) -> Tensor:
        B, _, S = inputs.size()
        index = self._batched_mapping[key]["full"][:B, :, :S]
        outputs = torch.gather(inputs, dim=1, index=index)
        return outputs

    def to(self, device=torch.device) -> None:  # type: ignore
        for k, v in self._mapping.items():
            self._mapping[k] = {x: y.to(device) for (x, y) in v.items()}  # type: ignore
        for k, v in self._batched_mapping.items():
            self._batched_mapping[k] = {x: y.to(device) for (x, y) in v.items()}  # type: ignore
        for k, v in self._batched_offset.items():
            self._batched_offset[k] = v.to(device)  # type: ignore

    def get_device(self) -> torch.device:
        return self._mapping["label"]["full"].device
