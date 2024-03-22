from typing import Optional

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor


def batch_topk_mask(
    scores: Tensor,
    topk: Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Given batched scores, return top-k larger masks and k-th index
    """
    B, S = scores.size()
    assert scores.ndim == 2 and topk.ndim == 1
    assert B == topk.size(0)
    assert 1 <= topk.min() and topk.max() <= S, topk
    if mask is not None:
        assert mask.size() == scores.size()
        assert (scores.size(1) >= topk).all()

    # ignore scores where mask = False by setting extreme values
    if mask is not None:
        const = torch.full_like(scores, fill_value=-1.0 * float("Inf"))
        scores = torch.where(mask, scores, const)

    sorted_values, _ = torch.sort(scores, dim=-1, descending=True)
    topk = torch.clamp(
        topk - 1, min=0
    )  # convert 1-indexed to 0-indexed and clip to avoid out-of-bound
    topk = rearrange(topk, "b -> b 1")

    k_th_scores = torch.gather(sorted_values, dim=1, index=topk)
    topk_mask = scores >= k_th_scores
    return topk_mask, k_th_scores


def sequence_mask(
    length: torch.LongTensor, maxlen: Optional[int] = None
) -> torch.BoolTensor:
    """
    Similar to https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    """
    B = length.size(0)
    maxlen = maxlen if maxlen else int(length.max())
    indices = repeat(torch.arange(maxlen), "s -> b s", b=B)
    mask = indices < rearrange(length, "b -> b 1")
    return mask.bool()  # type: ignore


def sample_mask(mask: torch.BoolTensor, ratio: Tensor) -> torch.Tensor:
    """
    Generate sampled_mask (B, S) given mask (B, S) according to the specified ratio
    If mask[b, s] is False, sampled_mask[b, s] should be False.
    """
    scores = torch.rand(mask.size())
    n_elem = reduce(mask, "b s -> b", reduction="sum")
    topk = (ratio * n_elem).long().clamp(min=1)
    sampled_mask, _ = batch_topk_mask(scores, topk, mask=mask)
    return sampled_mask


if __name__ == "__main__":
    B = 2
    sample_mask(torch.full((B, 3), fill_value=False), torch.full((B,), fill_value=0.5))  # type: ignore
