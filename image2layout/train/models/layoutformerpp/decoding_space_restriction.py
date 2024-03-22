import torch
from torch import Tensor


def restrict_reliable_label_or_size(
    sampling_idx: int,
    cond: Tensor,
    logits: Tensor,
    pad_id: int,
    eos_id: int,
    max_length: int,
) -> Tensor:

    B = cond.size(0)
    assert cond.size(1) == max_length + 1

    for batch_idx in range(B):
        given_cond: int = cond[batch_idx, sampling_idx].item()  # token
        # First pad position means <eos> token.
        first_pad_idx = torch.argmax((cond[batch_idx] == pad_id).float())
        first_pad_idx = (
            first_pad_idx.item()
            if cond[batch_idx, first_pad_idx] == pad_id
            else float("Inf")
        )

        mask = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
        if sampling_idx < first_pad_idx:
            # Mask out all tokens except for given_cond.
            if given_cond == pad_id or given_cond == -1:
                continue
            mask[given_cond] = False
        else:
            # Mask out all tokens except for eos token.
            mask[eos_id] = False

        logits[batch_idx, mask] = -float("Inf")

    return logits


def restrict_only_category(
    sampling_idx: int,
    cond: Tensor,
    logits: Tensor,
    pad_id: int,
    eos_id: int,
    max_length: int,
) -> Tensor:
    """
    In refinement, we need to restrict the label space.
    Spatial information is noisy so that we cannot use it.
    """

    if (sampling_idx - 1) % 5 != 0:
        return logits

    B = cond.size(0)
    assert cond.size(1) == max_length + 1
    assert B == logits.size(0)

    for batch_idx in range(B):
        given_cond: int = cond[batch_idx, sampling_idx].item()  # token
        # First pad position means <eos> token.
        first_pad_idx = torch.argmax((cond[batch_idx] == pad_id).float())
        first_pad_idx = (
            first_pad_idx.item()
            if cond[batch_idx, first_pad_idx] == pad_id
            else float("Inf")
        )

        mask = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
        if sampling_idx < first_pad_idx:
            # Mask out all tokens except for given_cond.
            if given_cond == pad_id or given_cond == -1:
                continue
            mask[given_cond] = False
        else:
            # Mask out all tokens except for eos token.
            mask[eos_id] = False

        logits[batch_idx, mask] = -float("Inf")

    return logits


def identiy_func(
    sampling_idx: int,
    cond: Tensor,
    logits: Tensor,
    pad_id: int,
    eos_id: int,
    max_length: int,
) -> Tensor:
    return logits


DECODE_SPACE_RESTRICTION = {
    "none": identiy_func,
    "uncond": identiy_func,
    "cwh": restrict_reliable_label_or_size,
    "c": restrict_reliable_label_or_size,
    "refinement": restrict_only_category,
    "partial": identiy_func,
    "relation": restrict_only_category,
}
