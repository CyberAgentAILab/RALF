# From https://github.com/microsoft/LayoutGeneration/blob/main/LayoutFormer%2B%2B/src/model/layout_transformer/constrained_decoding.py
# coding=utf8
import copy
import math
from math import ceil, floor
from typing import List, Set

import numpy as np
import torch
from image2layout.train.helpers.relationships import (
    REL_SIZE_ALPHA,
    RELATIVE_RELATION,
    RelElement,
    RelLoc,
    RelSize,
)
from image2layout.train.models.layoutformerpp.task_preprocessor import (
    RelationshipPreprocessor,
)
from torch import BoolTensor


def decapulate(bbox):
    if len(bbox.size()) == 2:
        x1, y1, x2, y2 = bbox.T
    else:
        x1, y1, x2, y2 = bbox.permute(2, 0, 1)
    return x1, y1, x2, y2


class RelationTypes:
    types = [
        RelSize.SMALLER,
        RelSize.EQUAL,
        RelSize.LARGER,
        RelSize.UNKNOWN,
        RelLoc.TOP,
        RelLoc.CENTER,
        RelLoc.BOTTOM,
        RelLoc.UNKNOWN,
        RelLoc.LEFT,
        RelLoc.RIGHT,
    ]
    _type2index = None
    _index2type = None

    @classmethod
    def type2index(self):
        if self._type2index is None:
            self._type2index = dict()
            for idx, type in enumerate(self.types):
                self._type2index[type] = idx
        return self._type2index

    @classmethod
    def index2type(self):
        if self._index2type is None:
            self._index2type = dict()
            for idx, type in enumerate(self.types):
                self._index2type[idx] = type
        return self._index2type


class DiscretizeBoundingBox:
    def __init__(self, num_x_grid: int, num_y_grid: int) -> None:
        self.num_x_grid = num_x_grid
        self.num_y_grid = num_y_grid
        self.max_x = self.num_x_grid - 1
        self.max_y = self.num_y_grid - 1

    def discretize(self, bbox):
        """
        Args:
            continuous_bbox torch.Tensor: N * 4
        Returns:
            discrete_bbox torch.LongTensor: N * 4
        """
        cliped_boxes = torch.clip(bbox, min=0.0, max=1.0)
        x1, y1, x2, y2 = decapulate(cliped_boxes)
        discrete_x1 = torch.floor(x1 * self.max_x)
        discrete_y1 = torch.floor(y1 * self.max_y)
        discrete_x2 = torch.floor(x2 * self.max_x)
        discrete_y2 = torch.floor(y2 * self.max_y)
        return torch.stack(
            [discrete_x1, discrete_y1, discrete_x2, discrete_y2], dim=-1
        ).long()

    def continuize(self, bbox):
        """
        Args:
            discrete_bbox torch.LongTensor: N * 4

        Returns:
            continuous_bbox torch.Tensor: N * 4
        """
        x1, y1, x2, y2 = decapulate(bbox)
        cx1, cx2 = x1 / self.max_x, x2 / self.max_x
        cy1, cy2 = y1 / self.max_y, y2 / self.max_y
        return torch.stack([cx1, cy1, cx2, cy2], dim=-1).float()

    def continuize_num(self, num: int) -> float:
        return num / self.max_x

    def discretize_num(self, num: float) -> int:
        return int(math.floor(num * self.max_y))

    def __call__(self, data):
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])
        discrete_bboxes = self.discretize(data["bboxes"])
        data["discrete_bboxes"] = discrete_bboxes
        discrete_gold_bboxes = self.discretize(data["gold_bboxes"])
        data["discrete_gold_bboxes"] = discrete_gold_bboxes
        return data


class TransformerSortByDictConstraintDecodeState:

    ELEMENT = "element"
    NUMBER = "number"
    SEP = "sep"

    def __init__(self, num_elements: int) -> None:
        self.num_elements = num_elements
        self.curr_element = 0
        self.next_token_type = self.ELEMENT
        self.num_bbox = 0
        self.pred_labels = list()
        self.pred_bbox = list()

    @property
    def finished(self):
        return self.curr_element == self.num_elements and self.num_bbox >= 4

    def add_label(self, label_token: str):
        self.pred_labels.append(label_token)
        self.pred_bbox.append(list())

    def add_bbox_num(self, num: str):
        self.pred_bbox[-1].append(int(num))

    def __repr__(self):
        return f"{self.num_elements=}\n{self.curr_element=}\n{self.num_bbox=}\n{self.pred_labels=}\n{self.pred_bbox=}"


class TransformerSortByDictLabelConstraint:
    def __init__(
        self,
        preprocessor: RelationshipPreprocessor,
    ) -> None:
        self.preprocessor = preprocessor
        self.discrete_degree = int(preprocessor.tokenizer._num_bin)

    @property
    def all_plausible_tokens(self):
        token_ids = self.special_token_ids | self.num_token_ids
        for label in self.label_token_ids.keys():
            token_ids |= self.label_token_ids[label]
        return token_ids

    def prepare(self, label_ids: List[List[int]]):
        self.constraints, self.decode_state = list(), list()
        for item_label_ids in label_ids:
            # lid 0 are pad labels
            if isinstance(item_label_ids, list):
                _item_label_ids = item_label_ids
            else:
                _item_label_ids = item_label_ids.tolist()
            item_label_names = [
                self.index2label[lid].lower().strip()
                for lid in _item_label_ids
                if lid > 0
            ]
            item_label_token_id = self.tokenizer.convert_tokens_to_ids(item_label_names)
            self.constraints.append(item_label_token_id)
            self.decode_state.append(
                [TransformerSortByDictConstraintDecodeState(len(item_label_names))]
            )

    def __call__(
        self, batch_id: int, seq_id: int, token_ids: torch.Tensor
    ) -> List[int]:

        label_constraints = self.constraints[batch_id]
        self.decode_state[batch_id] = self.decode_state[batch_id][: seq_id + 1]
        state = copy.deepcopy(self.decode_state[batch_id][-1])

        if state.finished:
            plausible_token_id = self.special_token_ids
        else:
            if (
                state.next_token_type
                == TransformerSortByDictConstraintDecodeState.ELEMENT
            ):
                plausible_token_id = {label_constraints[state.curr_element]}
                state.curr_element += 1
                state.next_token_type = (
                    TransformerSortByDictConstraintDecodeState.NUMBER
                )
                state.num_bbox = 0
            elif (
                state.next_token_type
                == TransformerSortByDictConstraintDecodeState.NUMBER
            ):
                plausible_token_id = self.num_token_ids
                state.num_bbox += 1
                if state.num_bbox >= 4:
                    if self.add_sep_token:
                        state.next_token_type = (
                            TransformerSortByDictConstraintDecodeState.SEP
                        )
                    else:
                        state.next_token_type = (
                            TransformerSortByDictConstraintDecodeState.ELEMENT
                        )
            else:
                # SEP
                plausible_token_id = self.label_suffix
                state.next_token_type = (
                    TransformerSortByDictConstraintDecodeState.ELEMENT
                )

            self.decode_state[batch_id].append(state)

        return list(plausible_token_id), None


class TransformerSortByDictLabelSizeConstraint(TransformerSortByDictLabelConstraint):
    def prepare(self, label_ids: List[List[int]], bboxes: List[List[int]]):
        self.label_constraints, self.size_constraints, self.decode_state = (
            list(),
            list(),
            list(),
        )
        for item_label_ids, item_bboxes in zip(label_ids, bboxes):
            # lid 0 are pad labels
            if isinstance(item_label_ids, list):
                _item_label_ids = item_label_ids
            else:
                _item_label_ids = item_label_ids.tolist()
            item_label_names = [
                self.index2label[lid].lower().strip()
                for lid in _item_label_ids
                if lid > 0
            ]
            item_label_token_id = self.tokenizer.convert_tokens_to_ids(item_label_names)

            if isinstance(item_bboxes, list):
                _item_bboxes = item_bboxes
            else:
                _item_bboxes = item_bboxes.tolist()
            item_size_token_id = [
                self.tokenizer.convert_tokens_to_ids(list(map(str, bbox[2:])))
                for bbox in _item_bboxes
            ]

            self.label_constraints.append(item_label_token_id)
            self.size_constraints.append(item_size_token_id)
            self.decode_state.append(
                [TransformerSortByDictConstraintDecodeState(len(item_label_names))]
            )

    def __call__(
        self, batch_id: int, seq_id: int, token_ids: torch.Tensor
    ) -> List[int]:

        label_constraints = self.label_constraints[batch_id]
        size_constraints = self.size_constraints[batch_id]
        self.decode_state[batch_id] = self.decode_state[batch_id][: seq_id + 1]
        state = copy.deepcopy(self.decode_state[batch_id][-1])

        if state.finished:
            plausible_token_id = self.special_token_ids
        else:
            if (
                state.next_token_type
                == TransformerSortByDictConstraintDecodeState.ELEMENT
            ):
                plausible_token_id = {label_constraints[state.curr_element]}
                state.curr_element += 1
                state.next_token_type = (
                    TransformerSortByDictConstraintDecodeState.NUMBER
                )
                state.num_bbox = 0
            elif (
                state.next_token_type
                == TransformerSortByDictConstraintDecodeState.NUMBER
            ):
                # size
                if state.num_bbox >= 2:
                    plausible_token_id = {
                        size_constraints[state.curr_element - 1][state.num_bbox - 2]
                    }
                else:
                    plausible_token_id = self.num_token_ids
                state.num_bbox += 1
                if state.num_bbox >= 4:
                    if self.add_sep_token:
                        state.next_token_type = (
                            TransformerSortByDictConstraintDecodeState.SEP
                        )
                    else:
                        state.next_token_type = (
                            TransformerSortByDictConstraintDecodeState.ELEMENT
                        )
            else:
                # SEP
                plausible_token_id = self.label_suffix
                state.next_token_type = (
                    TransformerSortByDictConstraintDecodeState.ELEMENT
                )
            self.decode_state[batch_id].append(state)

        return list(plausible_token_id), None


def detect_label_idx(types, label, index, map_relele_to_idx, preprocessor):
    indexes = torch.nonzero(types == label)
    idx = map_relele_to_idx[preprocessor.id_to_name(index.item())]
    label_pos = indexes[idx].item()
    return label_pos


def define_canvas_restriction(
    predictable_token_mask,
    tgt_ele_idx,
    total_bin,
):
    """
    Return:
        mask: True: Predictable, False: Unpredictable
    """
    # canvas constraint is generated by only cy.
    cy_start_idx = predictable_token_mask.nonzero()[0].item()
    assert cy_start_idx == 131
    rel_type = tgt_ele_idx
    mask = torch.zeros(
        predictable_token_mask.size(-1),
        dtype=torch.bool,
        device=predictable_token_mask.device,
    )
    if rel_type == RelLoc.TOP:
        mask[cy_start_idx : cy_start_idx + total_bin // 3] = True
    elif rel_type == RelLoc.CENTER:
        mask[cy_start_idx + total_bin // 3 : cy_start_idx + 2 * total_bin // 3] = True
    elif rel_type == RelLoc.BOTTOM:
        mask[cy_start_idx + 2 * total_bin // 3 : cy_start_idx + total_bin] = True
    else:
        raise ValueError(f"Unknown rel_type: {rel_type}")

    return mask


class TransformerSortByDictRelationConstraint(TransformerSortByDictLabelConstraint):
    def __init__(
        self,
        preprocessor: RelationshipPreprocessor,
    ) -> None:
        super().__init__(preprocessor)

        self.map_relele_to_idx = {e: _idx for _idx, e in enumerate(RelElement)}

        self.CURRENT_ELEM = {
            0: "Type",
            1: "Width",
            2: "Height",
            3: "Cx",
            4: "Cy",
        }

        self.ELEM_TO_ID = {v: k for k, v in self.CURRENT_ELEM.items()}

        self._token_mask = self.preprocessor.tokenizer.token_mask

        self.width_start_idx = self.token_mask[1].nonzero()[0].item()
        self.width_end_idx = self.token_mask[1].nonzero()[-3].item() + 1
        assert self.width_end_idx - self.width_start_idx == self.discrete_degree

        self.height_start_idx = self.token_mask[2].nonzero()[0].item()
        self.height_end_idx = self.token_mask[2].nonzero()[-3].item() + 1
        assert self.height_end_idx - self.height_start_idx == self.discrete_degree

        self.center_x_start_idx = self.token_mask[3].nonzero()[0].item()
        self.center_x_end_idx = self.token_mask[3].nonzero()[-3].item() + 1
        assert self.center_x_end_idx - self.center_x_start_idx == self.discrete_degree

        self.center_y_start_idx = self.token_mask[4].nonzero()[0].item()
        self.center_y_end_idx = self.token_mask[4].nonzero()[-3].item() + 1
        assert self.center_y_end_idx - self.center_y_start_idx == self.discrete_degree

        self.START_IDX = {
            "Width": self.width_start_idx,
            "Height": self.height_start_idx,
            "Cx": self.center_x_start_idx,
            "Cy": self.center_y_start_idx,
        }

        self.PREV_ELEM = {
            # "Width": "",
            "Height": "Width",
            "Cx": "Height",
            "Cy": "Cx",
            "Type": "Cy",
        }

    @property
    def token_mask(self) -> BoolTensor:
        return self._token_mask.clone()

    @property
    def logits_size(self) -> int:
        return self._token_mask.size(-1)

    @property
    def canvas_size(self) -> int:
        return self.discrete_degree - 1  # Size should be [0, 127]

    def prepare(self, _seq) -> list[list[torch.Tensor]]:
        """
        Args:
            _seq: Preprocesed sequence of relationships
        """
        _eos_idnex = torch.argmax((_seq == self.preprocessor.name_to_id("eos")).float())
        _relation_sep_index = torch.argmax(
            (_seq == self.preprocessor.name_to_id("relation_sep")).float()
        )

        _seq = _seq[:_eos_idnex]

        # Remove bos, task_tag, eot
        _types = _seq[3:_relation_sep_index][::2]
        self.type_constraint_token_id = _types
        _relations = _seq[_relation_sep_index + 1 :]
        _relations = _relations[_relations != self.preprocessor.name_to_id("sep")]
        a = np.array([self.preprocessor.id_to_name(e.item()) for e in _relations])
        _relations = _relations.reshape(-1, 5)  # [N, 5]

        NUM_ELEMENT = _types.size(0)
        self.decode_state = [TransformerSortByDictConstraintDecodeState(NUM_ELEMENT)]

        # Relation
        rel_constraints = [list() for _ in range(NUM_ELEMENT)]
        for kdx, rel in enumerate(_relations):
            _rel = np.array([self.preprocessor.id_to_name(e.item()) for e in rel])
            label_i, index_i, rel_type_idx, label_j, index_j = rel
            rel_type = self.preprocessor.id_to_name(rel_type_idx.item())

            # Detect label idx from parsed relation information (label_i)
            label_i_pos = detect_label_idx(
                _types, label_i, index_i, self.map_relele_to_idx, self.preprocessor
            )

            is_canvas = "canvas" == self.preprocessor.id_to_name(label_j.item())
            if is_canvas:
                rel_constraints[label_i_pos].append(
                    (
                        f"canvas",
                        rel_type,
                    )
                )
            else:
                # Detect label idx from parsed relation information (label_j)
                label_j_pos = detect_label_idx(
                    _types,
                    label_j,
                    index_j,
                    self.map_relele_to_idx,
                    self.preprocessor,
                )

                if label_j_pos > label_i_pos:
                    label_i, index_i, label_i_pos, label_j, index_j, label_j_pos = (
                        label_j,
                        index_j,
                        label_j_pos,
                        label_i,
                        index_i,
                        label_i_pos,
                    )
                    rel_type = RELATIVE_RELATION[rel_type]

                assert (
                    label_i_pos > label_j_pos
                ), f"{label_j_pos=}, {label_i_pos=}, {rel_type=}"
                rel_constraints[label_i_pos].append(
                    (
                        rel_type,
                        label_j_pos,
                    )
                )

        label_names = self.preprocessor.tokenizer._label_feature.names
        self.label_tokens = [
            self.preprocessor.name_to_id(_label) for _label in label_names
        ]
        self.special_tokens = [
            self.preprocessor.name_to_id(spe_token)
            for spe_token in self.preprocessor.tokenizer.special_tokens
        ]

        return rel_constraints

    def _intersect(self, a: Set, b: Set) -> Set:
        if len(b) == 0:
            return a
        intersection = a & b
        return intersection

    def plausible_tokens(self):
        return set(range(self.discrete_degree))

    def __call__(self, token_ids: torch.Tensor, rel_constraints: list) -> List[int]:

        batch_id = 0

        seq_len = token_ids.size(1) - 1  # -1 for eos
        self.decode_state = self.decode_state[: seq_len + 1]

        state = copy.deepcopy(self.decode_state[-1])
        current_elem = self.CURRENT_ELEM[seq_len % 5]

        if seq_len > 0:
            last_token = token_ids[batch_id, -1].item()
            if last_token in self.label_tokens:
                state.add_label(last_token)
            else:
                state.add_bbox_num(
                    last_token - self.START_IDX[self.PREV_ELEM[current_elem]]
                )  # BBox is aligned to [0, self.canvas_size]

        back_idx = None
        if state.finished:
            eos_idx = self.preprocessor.tokenizer.name_to_id("eos")
            mask = torch.ones(self.logits_size, dtype=torch.bool)
            mask[eos_idx] = False
        else:
            if current_elem == "Type":
                state.curr_element += 1
                state.next_token_type = (
                    TransformerSortByDictConstraintDecodeState.NUMBER
                )
                state.num_bbox = 0
                mask = torch.ones(self.logits_size, dtype=torch.bool)
                next_type_idx = seq_len // 5
                next_type_token_idx = self.type_constraint_token_id[next_type_idx]
                mask[next_type_token_idx] = False

            else:

                relation_constraints = rel_constraints[state.curr_element - 1]

                if state.curr_element == 1:

                    plausible_intersect = self.plausible_tokens()

                    for rdx, (rel_type, tgt_ele_idx) in enumerate(relation_constraints):

                        if current_elem == "Cy" and rel_type == "canvas":
                            # Canvas constraint is only implemented for Cy
                            _, curr_height, *_ = state.pred_bbox[-1]
                            rel_type = tgt_ele_idx
                            half_height = curr_height / 2
                            if rel_type == RelLoc.TOP:  # ok
                                min_h = ceil(half_height)
                                max_h = floor(self.canvas_size / 3 - half_height)
                            elif rel_type == RelLoc.CENTER:  # ok
                                min_h = ceil(1 * self.canvas_size / 3 + half_height)
                                max_h = floor(2 * self.canvas_size / 3 - half_height)
                            elif rel_type == RelLoc.BOTTOM:  # ok
                                min_h = ceil(2 * self.canvas_size / 3 + half_height)
                                max_h = floor(self.canvas_size - half_height)
                            else:
                                raise ValueError(f"Unknown rel_type: {rel_type}")
                            diff = set(range(min_h, max_h))

                            plausible_intersect = self._intersect(
                                plausible_intersect, diff
                            )

                    shifted_plausible_tokens = (
                        np.array(list(plausible_intersect))
                        + self.START_IDX[current_elem]
                    )
                    mask = torch.ones(self.logits_size, dtype=torch.bool)
                    mask[shifted_plausible_tokens] = False

                else:

                    if len(relation_constraints) == 0:
                        plausible_tokens = self.preprocessor.tokenizer.token_mask[
                            seq_len
                        ]
                        # True: Predictable, False: Unpredictable
                        mask = ~plausible_tokens

                    else:

                        plausible_intersect = self.plausible_tokens()

                        for rdx in range(len(relation_constraints)):

                            rel_type, tgt_ele_idx = relation_constraints[rdx]
                            is_canvas = rel_type == "canvas"
                            (
                                rel_type,
                                tgt_ele_bbox,
                                back_idx,
                            ) = self.get_target_bbox(rel_type, tgt_ele_idx, state)

                            if is_canvas and current_elem != "Cy":
                                continue

                            if current_elem == "Cx":

                                curr_width, _ = state.pred_bbox[-1]

                                tgt_w, _, tgt_cx, _ = tgt_ele_bbox

                                # Location
                                if rel_type == RelLoc.LEFT:  # ok?
                                    max_cx = ceil(self.canvas_size - curr_width / 2)
                                    min_cx = floor(tgt_cx + tgt_w / 2 + curr_width / 2)
                                    diff = set(range(min_cx, max_cx))
                                elif rel_type == RelLoc.RIGHT:  # ok?
                                    max_cx = ceil(tgt_cx - tgt_w / 2 - curr_width / 2)
                                    min_cx = floor(curr_width / 2)
                                    diff = set(range(min_cx, max_cx))
                                elif rel_type == RelLoc.CENTER:  # ok?
                                    max_cx = floor(tgt_cx + tgt_w / 2 - curr_width / 2)
                                    min_cx = ceil(tgt_cx - tgt_w / 2 + curr_width / 2)
                                    diff = set(range(min_cx, max_cx))
                                else:
                                    max_cx = ceil(self.canvas_size - curr_width / 2)
                                    min_cx = floor(curr_width / 2)
                                    diff = set(range(min_cx, max_cx))

                            elif current_elem == "Cy":

                                _, curr_height, *_ = state.pred_bbox[-1]

                                if is_canvas:

                                    # Canvas constraint is only implemented for Cy
                                    rel_type = tgt_ele_idx
                                    half_height = curr_height / 2
                                    if rel_type == RelLoc.TOP:  # ok
                                        min_h = ceil(half_height)
                                        max_h = floor(
                                            self.canvas_size / 3 - half_height
                                        )
                                    elif rel_type == RelLoc.CENTER:  # ok
                                        min_h = ceil(
                                            1 * self.canvas_size / 3 + half_height
                                        )
                                        max_h = floor(
                                            2 * self.canvas_size / 3 - half_height
                                        )
                                    elif rel_type == RelLoc.BOTTOM:  # ok
                                        min_h = ceil(
                                            2 * self.canvas_size / 3 + half_height
                                        )
                                        max_h = floor(self.canvas_size - half_height)
                                    else:
                                        raise ValueError(
                                            f"Unknown rel_type: {rel_type}"
                                        )
                                    diff = set(range(min_h, max_h))

                                else:

                                    _, tgt_h, _, tgt_cy = tgt_ele_bbox
                                    half_curr_h = curr_height / 2

                                    if rel_type == RelLoc.TOP:  # ok?
                                        max_cy = ceil(self.canvas_size - half_curr_h)
                                        min_cy = floor(tgt_cy + tgt_h / 2 + half_curr_h)
                                        diff = set(range(min_cy, max_cy))
                                    elif rel_type == RelLoc.BOTTOM:  # ok?
                                        max_cy = ceil(tgt_cy - tgt_h / 2 - half_curr_h)
                                        min_cy = floor(half_curr_h)
                                        diff = set(range(min_cy, max_cy))
                                    elif rel_type == RelLoc.CENTER:  # ok?
                                        max_cy = floor(tgt_cy + tgt_h / 2 + half_curr_h)
                                        min_cy = ceil(tgt_cy - tgt_h / 2 - half_curr_h)
                                        diff = set(range(min_cy, max_cy))
                                    else:
                                        max_cy = ceil(self.canvas_size - half_curr_h)
                                        min_cy = floor(curr_height / 2)
                                        diff = set(range(min_cy, max_cy))

                            elif current_elem == "Width":

                                tgt_w, tgt_h, tgt_cx, _ = tgt_ele_bbox
                                tgt_area = tgt_w * tgt_h

                                # Location
                                if rel_type == RelLoc.LEFT:  # ok?
                                    min_w = 0
                                    max_w = ceil(self.canvas_size - tgt_cx - tgt_w / 2)
                                    diff = set(range(min_w, max_w))
                                elif rel_type == RelLoc.RIGHT:  # ok?
                                    min_w = 0
                                    max_w = ceil(tgt_cx - tgt_w / 2)
                                    diff = set(range(min_w, max_w))
                                elif rel_type == RelLoc.CENTER:
                                    if tgt_cx < self.discrete_degree // 2:
                                        min_w = 0
                                        max_w = floor(
                                            self.canvas_size - tgt_cx + tgt_w / 2
                                        )
                                    else:
                                        min_w = 0
                                        max_w = floor(tgt_cx + tgt_w / 2)
                                    diff = set(range(min_w, max_w))
                                # Size
                                elif rel_type == RelSize.SMALLER:  # ok?
                                    tgt_area /= 1 - REL_SIZE_ALPHA
                                    min_w = min(
                                        ceil(tgt_area / self.canvas_size),
                                        self.canvas_size,
                                    )
                                    max_w = ceil(tgt_area)
                                    diff = set(range(min_w, max_w))
                                elif rel_type == RelSize.LARGER:  # ok?
                                    tgt_area /= 1 + REL_SIZE_ALPHA
                                    min_w = 0
                                    max_w = floor(tgt_area / self.canvas_size)
                                    diff = set(range(min_w, max_w))
                                elif rel_type == RelSize.EQUAL:
                                    # MIN
                                    min_tgt_area = tgt_area / (1 + REL_SIZE_ALPHA)
                                    min_w = floor(min_tgt_area / self.canvas_size)
                                    # MAX
                                    max_tgt_area = tgt_area / (1 - REL_SIZE_ALPHA)
                                    max_w = ceil(max_tgt_area / self.canvas_size)
                                    diff = set(range(min_w, max_w))
                                else:
                                    diff = self.plausible_tokens()

                            elif current_elem == "Height":

                                curr_width = state.pred_bbox[-1][0]

                                _, tgt_h, _, tgt_cy = tgt_ele_bbox
                                tgt_area = tgt_ele_bbox[0] * tgt_h

                                if rel_type == RelLoc.TOP:  # ok?
                                    min_h = 0
                                    max_h = ceil(tgt_cy - tgt_h / 2)
                                    diff = set(range(min_h, max_h))
                                elif rel_type == RelLoc.BOTTOM:  # ok?
                                    min_h = 0
                                    max_h = floor(tgt_cy - tgt_h / 2)  # floor?
                                    diff = set(range(min_h, max_h))
                                elif rel_type == RelLoc.CENTER:  # ok?
                                    min_h = 0
                                    if tgt_cy < self.discrete_degree // 2:
                                        max_h = floor(
                                            self.canvas_size - tgt_cy + tgt_h / 2
                                        )
                                    else:
                                        max_h = floor(tgt_cy + tgt_h / 2)
                                    diff = set(range(min_h, max_h))
                                # Size
                                elif rel_type == RelSize.SMALLER:
                                    tgt_area /= 1 - REL_SIZE_ALPHA
                                    if curr_width == 0:
                                        min_h = self.canvas_size
                                    else:
                                        min_h = min(
                                            ceil(tgt_area / curr_width),
                                            self.canvas_size,
                                        )
                                    max_h = self.discrete_degree
                                    diff = set(range(min_h, max_h))
                                elif rel_type == RelSize.LARGER:
                                    tgt_area /= 1 + REL_SIZE_ALPHA
                                    min_h = 0
                                    if curr_width == 0:
                                        max_h = self.discrete_degree
                                    else:
                                        max_h = min(
                                            floor(tgt_area / curr_width),
                                            self.discrete_degree,
                                        )
                                    diff = set(range(min_h, max_h))
                                elif rel_type == RelSize.EQUAL:
                                    # MIN
                                    min_tgt_area = tgt_area / (1 + REL_SIZE_ALPHA)
                                    if curr_width == 0:
                                        curr_width = 1
                                    min_w = floor(min_tgt_area / curr_width)
                                    # MAX
                                    max_tgt_area = tgt_area / (1 - REL_SIZE_ALPHA)
                                    max_w = ceil(max_tgt_area / curr_width)
                                    diff = set(range(min_w, max_w))
                                else:
                                    diff = self.plausible_tokens()

                            plausible_intersect = self._intersect(
                                plausible_intersect, diff
                            )

                        shifted_plausible_tokens = (
                            np.array(list(plausible_intersect))
                            + self.START_IDX[current_elem]
                        )
                        mask = torch.ones(self.logits_size, dtype=torch.bool)
                        mask[shifted_plausible_tokens] = False

                state.num_bbox += 1
            self.decode_state.append(state)
        return mask, back_idx

    def get_target_bbox(self, rel_type, tgt_ele_idx, state):
        back_idx = None
        is_canvas = rel_type == "canvas"
        if is_canvas:
            rel_type = tgt_ele_idx
            return rel_type, None, back_idx
        else:
            if tgt_ele_idx is None:
                return rel_type, [0, 0, self.canvas_size, self.canvas_size], back_idx
            back_idx = tgt_ele_idx * 5 + state.num_bbox + 1
            return rel_type, state.pred_bbox[tgt_ele_idx], back_idx
