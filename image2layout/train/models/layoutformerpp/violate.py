from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import seaborn as sns
import torch
from torch import Tensor

from ...helpers.layout_tokenizer import LayoutSequenceTokenizer
from ...helpers.relationships import (
    RelLoc,
    RelSize,
    convert_xywh_to_ltrb,
    detect_loc_relation_between_elements,
    detect_size_relation,
)
from ..common.base_model import (
    ConditionalInputsForDiscreteLayout,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)


def calculate_violation(
    cond_type: str,
    cond: Union[
        ConditionalInputsForDiscreteLayout,
        RetrievalAugmentedConditionalInputsForDiscreteLayout,
    ],
    output: dict[str, Tensor],
    output_seq: torch.Tensor,
    tokenizer: LayoutSequenceTokenizer,
    prepared_rel_constraints: list,
) -> dict[str, float]:
    """
    Args:
        cond_type: "c" / "cwh" / "partial" / "relation"
    """

    # Remove <sos> token
    if cond_type == "uncond" or cond_type == "none":
        return empty_vio_rate()

    if cond_type == "c":
        cal_func = calculate_vio_rate_reliable_label_or_size
    elif cond_type == "cwh":
        cal_func = calculate_vio_rate_reliable_label_or_size
    elif cond_type == "partial":
        cal_func = empty_vio_rate
    elif cond_type == "refinement":
        cal_func = calculate_vio_rate_reliable_label_or_size
    elif cond_type == "relation":
        return calculate_vio_rate_relation(
            cond,
            output_seq,
            prepared_rel_constraints,
        )
    else:
        raise ValueError(f"Unknown cond_type: {cond_type}")

    return cal_func(
        cond_type,
        cond,
        output,
        tokenizer,
    )


def remove_unncecessary_tokens(
    seq,
    pad_mask,
    pad_id,
    eos_id,
):
    seq = seq[pad_mask]
    seq = seq[seq != pad_id]
    seq = seq[seq != eos_id]
    return seq


def empty_vio_rate(
    *args,
    **kwargs,
) -> dict[str, float]:
    return {
        "total": 1,
        "viorated": 0,
    }


def calculate_vio_rate_reliable_label_or_size(
    cond_type: str,
    cond,
    output: torch.Tensor,
    tokenizer: LayoutSequenceTokenizer,
):
    eos_id = tokenizer.name_to_id("eos")
    pad_id = tokenizer.name_to_id("pad")
    B = cond.seq.size(0)
    total_count = 0
    viorated_count = 0

    _seq = cond.seq[:, 1:].cpu()
    _mask = cond.mask[:, 1:].cpu()

    for batch_idx in range(B):

        _input_seq = _seq[batch_idx]
        _input_mask = _mask[batch_idx]
        _input_seq = remove_unncecessary_tokens(
            seq=_input_seq,
            pad_mask=_input_mask,
            pad_id=pad_id,
            eos_id=eos_id,
        ).cpu()

        _output_seq = output[batch_idx]

        if cond_type == "refinement":
            _output_seq = _output_seq[: _input_seq.size(0)][::5]
            _input_seq = _input_seq[::5]
        else:
            _output_seq = remove_unncecessary_tokens(
                seq=_output_seq,
                pad_mask=_input_mask,
                pad_id=pad_id,
                eos_id=eos_id,
            ).cpu()

        diff_elems = _input_seq.size(0) - _output_seq.size(0)
        assert diff_elems == 0, "diff_elems should be 0"

        viorated_count += int(torch.ne(_input_seq, _output_seq).sum().item())
        total_count += _input_seq.size(0)

    violation = {
        "total": total_count,
        "viorated": viorated_count,
    }
    return violation


def calculate_vio_rate_relation(
    cond,
    output: dict[str, Tensor],
    prepared_rel_constraints,
) -> dict[str, float]:

    B = cond.seq.size(0)
    assert len(prepared_rel_constraints) == B

    const_size = [RelSize.SMALLER, RelSize.EQUAL, RelSize.LARGER, RelSize.UNKNOWN]
    const_loc = [RelLoc.LEFT, RelLoc.TOP, RelLoc.RIGHT, RelLoc.BOTTOM, RelLoc.CENTER]

    TOTAL = 0
    VIORATED = 0
    PALETTE = sns.color_palette("deep", 10)

    for batch_idx in range(B):

        H = 750
        W = 513
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        relation_consts = prepared_rel_constraints[batch_idx]
        viorated_count = 0

        for i in range(len(relation_consts)):

            bbox_A = [
                output["center_x"][batch_idx, i].item(),
                output["center_y"][batch_idx, i].item(),
                output["width"][batch_idx, i].item(),
                output["height"][batch_idx, i].item(),
            ]
            lrrb_A = convert_xywh_to_ltrb(bbox_A)
            C = [int(c * 255.0) for c in PALETTE[i]]
            cv2.rectangle(
                canvas,
                (int(lrrb_A[0] * W), int(lrrb_A[1] * H)),
                (int(lrrb_A[2] * W), int(lrrb_A[3] * H)),
                C,
                2,
            )

            rel_const = relation_consts[i]
            if len(rel_const) == 0:
                continue

            for const in rel_const:

                TOTAL += 1

                rel_type = const[0]
                is_canvas = rel_type == "canvas"

                if is_canvas:
                    rel_type = const[1]
                    yc = bbox_A[1]

                    if yc < 1.0 / 3:
                        detected_rel = RelLoc.TOP
                    elif yc < 2.0 / 3:
                        detected_rel = RelLoc.CENTER
                    else:
                        detected_rel = RelLoc.BOTTOM

                    if detected_rel != rel_type:
                        VIORATED += 1
                    del yc

                else:
                    bbox_B = [
                        output["center_x"][batch_idx, const[1]].item(),
                        output["center_y"][batch_idx, const[1]].item(),
                        output["width"][batch_idx, const[1]].item(),
                        output["height"][batch_idx, const[1]].item(),
                    ]

                    if rel_type in const_size:
                        detected_rel = detect_size_relation(bbox_A, bbox_B)
                    elif rel_type in const_loc:
                        detected_rel = detect_loc_relation_between_elements(
                            bbox_A, bbox_B
                        )

                    if detected_rel != rel_type:
                        viorated_count += 1

        VIORATED += viorated_count

    violation = {
        "total": TOTAL,
        "viorated": VIORATED,
    }

    return violation
