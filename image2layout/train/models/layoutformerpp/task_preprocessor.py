import copy
import logging
import random
from abc import ABC
from dataclasses import dataclass
from typing import Any, Union

import fsspec
import torch
from einops import rearrange
from image2layout.train.global_variables import PRECOMPUTED_WEIGHT_DIR
from image2layout.train.models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)
from torch import BoolTensor, Tensor

from ...helpers.layout_tokenizer import LayoutTokenizer
from ...helpers.relationships import RelElement, RelLoc, RelSize
from ...helpers.task import VARS

logger = logging.getLogger(__name__)

TASK_TOKEN_VOCABULARIES = [
    "end_of_task",
    "label",
    "label_size",
    "relationship",
    "refinement",
    "completion",
    "uncondition",
]
SPECIAL_TOKEN_VOCABULARIES = ["sep", "relation_sep", "canvas"]

RELATIONSHIP_POSITION_VOCABULARIES = list(RelLoc)
RELATIONSHIP_SIZE_VOCABULARIES = list(RelSize)
RELATIONSHIP_UNIQUE_ELEMENT_VOCAVULARIES = list(RelElement)


@dataclass
class TaskPreprocessorOutput:
    """
    Output class for task preprocessor.

    Args:
        seq (`Tenspr`): [B, N]
            [[<sos>, TASK TOKEN, END OF TASK TOKEN, ...],..]
        pad_mask (`pad_mask`): [B, N]
    """

    seq: Tensor
    pad_mask: Tensor


class BasePreprocessor(ABC):
    def __init__(
        self,
        tokenizer: LayoutTokenizer,
        global_task_embedding: bool,
    ) -> None:

        self.tokenizer = tokenizer
        self.global_task_embedding = global_task_embedding

        self._RELATIONSHIP_TYPE_VOCABULARIES = [
            RELATIONSHIP_UNIQUE_ELEMENT_VOCAVULARIES[i]
            for i in range(self.tokenizer.max_seq_length)
        ]  # [A, B, C,...]

        self._preprocess_token_name_to_id = {
            token: self.tokens.index(token) + self.N_tokenizer_total
            for token in self.tokens
        }
        _labelname_to_id = {
            name: self.tokenizer._label_feature.names.index(name)
            for name in self.tokenizer._label_feature.names
        }
        self._token_to_name_to_id = {
            **self.tokenizer._special_token_name_to_id,
            **self._preprocess_token_name_to_id,
            **_labelname_to_id,
        }

        self._token_to_id_to_name = {
            v: k for (k, v) in self._token_to_name_to_id.items()
        }

        # note: these variables should be overridden
        self._TASK = ""
        self._VAR = ""

    def __call__(self):
        """
        Returns:
            seq (B, variable length)
            pad_mask (B, variable length)
                False means "valid" token.
                True means "pad" token.
        """
        raise NotImplementedError

    @property
    def var_order(self) -> list[str]:
        return self.tokenizer.var_order

    @property
    def tokens(self) -> list[str]:
        return self.task_tokens + self.special_tokens + self.relationshop_tokens

    @property
    def task_tokens(self) -> list[str]:
        return TASK_TOKEN_VOCABULARIES

    @property
    def special_tokens(self) -> list[str]:
        return SPECIAL_TOKEN_VOCABULARIES

    @property
    def relationshop_tokens(self) -> list[str]:
        return (
            self._RELATIONSHIP_TYPE_VOCABULARIES
            + RELATIONSHIP_POSITION_VOCABULARIES
            + RELATIONSHIP_SIZE_VOCABULARIES
        )  # type: ignore

    @property
    def N_tokenizer_total(self) -> int:
        return self.tokenizer.N_total

    @property
    def N_total(self) -> int:
        return self.tokenizer.N_total + len(self.tokens)

    @property
    def TASK(self) -> str:
        return self._TASK  # type: ignore

    def name_to_id(self, name: str) -> int:
        return self._token_to_name_to_id[name]  # type: ignore

    def id_to_name(self, id: int) -> str:
        return self._token_to_id_to_name[id]  # type: ignore

    def get_token(self, name: str, batch_size: int) -> Tensor:
        return torch.full((batch_size, 1), self.name_to_id(name)).to(
            self.device
        )  # [B, 1]

    def parse_seq_into_vars(
        self, seq: Tensor, shuffle_element: bool = True
    ) -> dict[str, Tensor]:
        """
        Args:
            seq: [B, 5*max_elem]
        Return:
            seq: [B, 5, max_elem]
                "5" means
                    - label
                    - center_x
                    - center_y
                    - width
                    - height
        """
        seq[seq == self.name_to_id("eos")] = self.name_to_id("pad")
        seq = seq[:, 1:]  # Remove bos token, [B, 5*max_elem]
        seq = seq.reshape(seq.size(0), -1, 5)  # .T
        seq = seq.permute(*torch.arange(seq.ndim - 1, -1, -1))
        seq = rearrange(seq, "c n b -> b c n")  # [B, 5, 10]
        _var_order = self.tokenizer.var_order

        if shuffle_element:

            B, C, N = seq.shape
            non_padding_mask = seq != self.name_to_id("pad")
            non_padding_counts = non_padding_mask.sum(dim=2)[..., 0]
            rand_indexes_list = [torch.randperm(x) for x in non_padding_counts]

            x_shuffled = seq.clone()
            for i in range(B):
                x_shuffled[i, :, : non_padding_counts[i]] = seq[
                    i, :, rand_indexes_list[i]
                ]
            seq = x_shuffled

        return {
            _var_order[0]: seq[:, 0],
            _var_order[1]: seq[:, 1],
            _var_order[2]: seq[:, 2],
            _var_order[3]: seq[:, 3],
            _var_order[4]: seq[:, 4],
        }

    def parse_mask(self, mask: Tensor) -> Tensor:
        """
        Args:
            mask: [B, 5*max_elem]
        Return:
            seq: [B, max_elem]
        """
        return self.parse_seq_into_vars(mask)["label"]

    @property
    def total_element_length(self) -> int:
        # MAX_SIZE: (label + height...) * valid elements - "1"
        # "1" denotes the final pad
        return (len(self._VAR) + 1) * self.NUM_VALID_ELEMENTS.max().item() - 1

    @property
    def total_sequence_length(self) -> Tensor:
        # MAX_SIZE: (label + height...) * valid elements - "1"
        # "1" denotes the final pad
        _num_elements: Tensor = len(self._VAR) * self.NUM_VALID_ELEMENTS
        NUM_BOS = 1
        NUM_EOS = 1
        if not self.global_task_embedding:
            NUM_TASK = 1
            NUM_END_OF_TASK = 1
        else:
            NUM_TASK = 0
            NUM_END_OF_TASK = 0
        NUM_SPE_TOKEN = NUM_BOS + NUM_EOS + NUM_TASK + NUM_END_OF_TASK

        # Calculate the number of "sep" i.e. "|".
        # If ```cwh | cwh | cwh```, then NUM_SEP=2
        _num_sep_token = torch.div(
            _num_elements - 1, len(self._VAR), rounding_mode="floor"
        )
        return NUM_SPE_TOKEN + _num_elements + _num_sep_token

    def set_seq_info(self, seq: dict[str, Tensor]) -> None:
        """
        Variables:
            NUM_VALID_ELEMENTS (Tensor):
                The number of valid elements in each batch.
            MAX_VALID_ELEMENT_POS (int):
                The maximum valid element position across batch.
            TOTAL_ELEMENT_SIZE (int):
                The maximum number of elements across batch.
            TOTAL_SEQ_SIZES (Tensor):
                The sequence size of each batch.
            PAD_SIZES (Tensor):
                The number of padding tokens in each batch.
        """

        _pad_id = self.name_to_id("pad")
        _eos_id = self.name_to_id("eos")
        _mask = (seq["label"] != _pad_id) & (seq["label"] != _eos_id)
        _mask_cumsum = _mask.cumsum(dim=1)[:, -1]

        self.device = seq["label"].device
        self.B = seq["label"].size(0)

        # Max index of mask.
        self.NUM_VALID_ELEMENTS: Tensor = _mask_cumsum
        self.MAX_VALID_ELEMENT_POS: int = self.NUM_VALID_ELEMENTS.max().item()
        self.TOTAL_ELEMENT_SIZE: int = self.total_element_length
        self.TOTAL_SEQ_SIZES: Tensor = self.total_sequence_length
        self.PAD_SIZES: Tensor = self.TOTAL_SEQ_SIZES.max() - self.TOTAL_SEQ_SIZES

    def insert_sep_token_between_elements(self, seq: dict[str, Tensor]) -> Tensor:
        # ラベル間にsep_tokenを挿入
        B = seq[self._VAR[0]].size(0)
        _sep_token = self.get_token("sep", B)  # [B, 1]
        expanded_labels_with_sep = torch.stack(
            [
                *[seq[_key][:, : self.MAX_VALID_ELEMENT_POS] for _key in self._VAR],
                _sep_token.repeat(1, self.MAX_VALID_ELEMENT_POS),
            ],
            dim=2,
        )
        expanded_labels_with_sep = expanded_labels_with_sep.view(B, -1)
        expanded_labels_with_sep = expanded_labels_with_sep[
            :, :-1
        ]  # Remove final sep_token
        assert expanded_labels_with_sep.size(1) == self.TOTAL_ELEMENT_SIZE

        return expanded_labels_with_sep

    # 不要なsep_tokenを取り除く
    def remove_unnecessary_sep_token(self, seq: Tensor) -> Tensor:
        B = seq.size(0)
        _device = seq.device
        valid_masks = (
            torch.arange(self.TOTAL_ELEMENT_SIZE, device=_device)
            .unsqueeze(0)
            .repeat(B, 1)
        )
        valid_masks = valid_masks < (self.TOTAL_SEQ_SIZES.unsqueeze(-1) - 2)
        expanded_labels = torch.where(
            valid_masks,
            seq,
            self.get_token("pad", B).repeat(1, self.TOTAL_ELEMENT_SIZE),
        )
        # expanded_labels[0]: [1, 6, 1, 6, 3, 3...] 6: sep_token, 1: cls label, 3: pad_token
        assert expanded_labels.size(1) == self.TOTAL_ELEMENT_SIZE
        return expanded_labels

    def create_task_token(self, batch_size: int) -> Tensor:
        _task = self.get_token(self.TASK, batch_size)
        _eot = self.get_token("end_of_task", batch_size)
        task_token = torch.cat([_task, _eot], dim=-1)
        return task_token

    def add_task_token(self, seq: Tensor) -> Tensor:

        B = seq.size(0)

        bos_token = self.get_token("bos", B)  # [B, 1]

        if not self.global_task_embedding:
            task_token = self.create_task_token(B)  # [B, 2]
            seq = torch.cat([bos_token, task_token, seq], dim=1)
        else:
            seq = torch.cat([bos_token, seq], dim=1)

        return seq

    def adjust_padding(self, seq: Tensor) -> Tensor:
        B = seq.size(0)

        PAD = self.get_token("pad", B)  # [B, 1]
        # 141="sep"

        seq = torch.cat([seq, PAD], dim=1)
        seq.scatter_(1, self.TOTAL_SEQ_SIZES.unsqueeze(-1) - 1, self.name_to_id("eos"))

        pad_counts = (seq == self.name_to_id("pad")).sum(dim=1)
        assert (pad_counts - self.PAD_SIZES).sum().item() == 0

        return seq

    def create_pad_mask(self, seq: Tensor) -> BoolTensor:
        pad_mask = torch.where(
            seq == self.name_to_id("pad"),
            torch.full_like(seq, fill_value=True, dtype=torch.bool),
            torch.full_like(seq, fill_value=False, dtype=torch.bool),
        )
        return pad_mask

    def decode_tokens(self, seq: Tensor) -> list:
        # seq: [bs, seq]

        _id_to_name = {i: str(i) for i in range(self.N_total)}
        for k, v in self._token_to_id_to_name.items():
            _id_to_name[k] = v
        output = []
        for _seq in seq:
            _seq = _seq.tolist()
            _seq = [_id_to_name[_token] for _token in _seq]
            output.append(_seq)
        return output


class UnconditionalPreprocessor(BasePreprocessor):
    """Unconstrained generation (UGen)"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._TASK = "uncondition"

    def __call__(
        self,
        inputs: Union[
            ConditionalInputsForDiscreteLayout,
            RetrievalAugmentedConditionalInputsForDiscreteLayout,
        ],
    ) -> dict[str, Tensor]:

        B = inputs.image.size(0)
        self.device = inputs.image.device

        bos = self.get_token("bos", B)  # [B, 1]
        # [B, 2], task_token + end_of_task_token
        eos = self.get_token("eos", B)  # [B, 1]
        if not self.global_task_embedding:
            task = self.create_task_token(B)
            seq: Tensor = torch.cat([bos, task, eos], dim=-1)
        else:
            seq: Tensor = torch.cat([bos, eos], dim=-1)

        pad_mask: Tensor = self.create_pad_mask(seq)

        return {"seq": seq, "pad_mask": pad_mask}


class BaseGeoPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(
        self,
        inputs: Union[
            ConditionalInputsForDiscreteLayout,
            RetrievalAugmentedConditionalInputsForDiscreteLayout,
        ],
        shuffle_element: bool = True,
    ) -> dict[str, Tensor]:

        seq: dict[str, Tensor] = self.parse_seq_into_vars(
            inputs.seq,
            shuffle_element,
        )  # [B, 5, max_elem]
        self.set_seq_info(seq)

        seq: Tensor = self.insert_sep_token_between_elements(seq)
        seq: Tensor = self.remove_unnecessary_sep_token(seq)
        seq: Tensor = self.add_task_token(seq)
        seq: Tensor = self.adjust_padding(seq)

        pad_mask: Tensor = self.create_pad_mask(seq)

        return {"seq": seq, "pad_mask": pad_mask}


class LabelPreprocessor(BaseGeoPreprocessor):
    """Generation conditioned on types (Gen-T)"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._TASK = "label"
        self._VAR = VARS["c"]


class LabelSizePreprocessor(BaseGeoPreprocessor):
    """Generation conditioned on types and sizes (Gen-TS)"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._TASK = "label_size"
        self._VAR = VARS["cwh"]

    def __call__(
        self,
        inputs: Union[
            ConditionalInputsForDiscreteLayout,
            RetrievalAugmentedConditionalInputsForDiscreteLayout,
        ],
    ) -> dict[str, Tensor]:
        # Replace "-1" with "pad" token.
        assert inputs.task == "cwh", ValueError(f"task={inputs.task}")
        return super().__call__(inputs=inputs, shuffle_element=False)


class RefinementPreprocessor(BaseGeoPreprocessor):
    """Refinement"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._TASK = "refinement"
        self._VAR = VARS["refinement"]

    def __call__(
        self,
        inputs: Union[
            ConditionalInputsForDiscreteLayout,
            RetrievalAugmentedConditionalInputsForDiscreteLayout,
        ],
    ) -> dict[str, Tensor]:
        # Replace "-1" with "pad" token.
        assert inputs.task == "refinement", ValueError(f"task={inputs.task}")
        return super().__call__(inputs=inputs, shuffle_element=False)


class PartialPreprocessor(BaseGeoPreprocessor):
    """Partial i.e. Completion"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._TASK = "completion"
        self._VAR = VARS["partial"]

    def __call__(
        self,
        inputs: Union[
            ConditionalInputsForDiscreteLayout,
            RetrievalAugmentedConditionalInputsForDiscreteLayout,
        ],
    ) -> dict[str, Tensor]:
        # Replace "-1" with "pad" token.
        assert inputs.task == "partial", ValueError(f"task={inputs.task}")
        assert list(set(inputs.seq[~inputs.mask].tolist())) == [-1]
        inputs = copy.deepcopy(inputs)
        inputs.seq[~inputs.mask] = self.name_to_id("pad")
        # Shuffle the order of elements.
        return super().__call__(inputs=inputs, shuffle_element=True)


class RelationshipPreprocessor(BasePreprocessor):
    """Generation conditioned on relationships (Gen-R)"""

    def __init__(self, RELATION_SIZE: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._TASK = "relationship"
        self._VAR = VARS["relation"]

        self.RELATION_SIZE = RELATION_SIZE

        relationship_table_path = (
            "cache/pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt"
        )
        fs, _ = fsspec.core.url_to_fs(relationship_table_path)
        if not fs.exists(relationship_table_path):
            relationship_table_path = f"{PRECOMPUTED_WEIGHT_DIR}/relationship/pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt"

        logger.info(f"Load relationship cache from {relationship_table_path}")
        self.table: dict[str, list] = torch.load(relationship_table_path)
        self.table = {k: random.sample(v, len(v)) for k, v in self.table.items()}

        self.label_preprocessor = LabelPreprocessor(
            tokenizer=kwargs["tokenizer"],
            global_task_embedding=self.global_task_embedding,
        )

    def set_relation_size(self, RELATION_SIZE: int) -> None:
        self.RELATION_SIZE = RELATION_SIZE

    def __call__(
        self,
        inputs: Union[
            ConditionalInputsForDiscreteLayout,
            RetrievalAugmentedConditionalInputsForDiscreteLayout,
        ],
    ) -> dict[str, Tensor]:

        # # Retrieve pre-computed relationships.
        if isinstance(inputs.id, Tensor):
            ids = inputs.id.cpu().tolist()
        else:
            ids = inputs.id
        relations = [self.table[str(_id)] for _id in ids]

        # Setup
        seq: dict[str, Tensor] = self.parse_seq_into_vars(
            inputs.seq
        )  # [B, 5, max_elem]
        self.set_seq_info(seq)

        seq_l = self.label_preprocessor(inputs)
        seq_label = seq_l["seq"]
        seq_label_mask = seq_l["pad_mask"]

        # Replace "label" token with "relationship" token.
        if not self.global_task_embedding:
            rel_task_token = self.get_token(self.TASK, self.B)
            seq_label[:, 1] = rel_task_token[:, 0]
        else:
            assert not self.name_to_id("label") in list(
                set(seq_label.flatten().tolist())
            )

        # Replace "eos" token with "relation_sep" token.
        seq_label[seq_label == self.name_to_id("eos")] = self.name_to_id("relation_sep")

        seq_relation = []  # dynamic length depends on a maximum `_sample_size``.
        MAXIMUM_LENGTH = -1
        for batch_idx in range(self.B):
            _seq = seq_label[batch_idx][~seq_label_mask[batch_idx]]

            # Random sample a 10% of relationship.
            _sample_size = max(len(relations[batch_idx]) * self.RELATION_SIZE // 100, 1)

            if len(relations[batch_idx]) == 0:
                # Add "eos" token.
                _seq = torch.cat([_seq, self.get_token("eos", 1)[0]], dim=0)
                seq_relation.append(_seq)
                continue

            # Equivalent to "shuffle_element=True"
            _relation = random.sample(
                relations[batch_idx],
                _sample_size,
            )
            # _relation = relations[batch_idx]  # TODO: fix

            # e.g. [['text', <RelElement.A: 10>, <RelSize.LARGER: 3>, 'underlay', <RelElement.B: 11>],...]
            _relation_tokenized = torch.tensor(
                [[self.name_to_id(_elem) for _elem in _rel] for _rel in _relation]
            ).to(self.device)

            _sep_token = self.get_token("sep", _relation_tokenized.size(0))
            _relation_tokenized_with_sep = torch.cat(
                [_relation_tokenized, _sep_token], dim=1
            )
            _relation_tokenized_with_sep = _relation_tokenized_with_sep.view(-1)
            _relation_tokenized_with_sep[-1] = self.name_to_id("eos")

            _seq_relation = torch.cat([_seq, _relation_tokenized_with_sep], dim=0)
            seq_relation.append(_seq_relation)

            MAXIMUM_LENGTH = max(MAXIMUM_LENGTH, _seq_relation.size(0))

        seq_out = torch.full(
            (self.B, MAXIMUM_LENGTH),
            fill_value=self.name_to_id("pad"),
            dtype=torch.long,
            device=self.device,
        )
        for batch_idx, _seq_relation in enumerate(seq_relation):
            if _seq_relation.size(0) == MAXIMUM_LENGTH:
                seq_out[batch_idx] = _seq_relation
            else:
                seq_out[batch_idx, : _seq_relation.size(0)] = _seq_relation

        # True if "pad" token.
        pad_mask = torch.where(
            seq_out == self.name_to_id("pad"),
            torch.ones_like(seq_out, dtype=torch.bool),
            torch.zeros_like(seq_out, dtype=torch.bool),
        )
        return {"seq": seq_out, "pad_mask": pad_mask}


PREPROCESSOR = {
    None: UnconditionalPreprocessor,
    "none": UnconditionalPreprocessor,
    "uncond": UnconditionalPreprocessor,
    "c": LabelPreprocessor,
    "cwh": LabelSizePreprocessor,
    "partial": PartialPreprocessor,
    "refinement": RefinementPreprocessor,
    "relation": RelationshipPreprocessor,
}
