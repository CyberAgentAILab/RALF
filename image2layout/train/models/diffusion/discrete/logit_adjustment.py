import logging

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from image2layout.train.global_variables import GEO_KEYS
from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from image2layout.train.helpers.relationships import RelLoc, RelSize
from image2layout.train.models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
)
from image2layout.train.models.diffusion.discrete.clg_lo import Graph
from image2layout.train.models.diffusion.discrete.clg_lo import (
    relation as relational_constraints,
)
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)

rel_unk = (1 << RelSize.UNKNOWN) | (1 << RelLoc.UNKNOWN)


def _index_to_smoothed_log_onehot(
    seq: torch.LongTensor,
    tokenizer: LayoutSequenceTokenizer,
    mode: str = "uniform",
    offset_ratio: float = 0.2,
) -> Tensor:
    # for ease of hp-tuning, the range is limited to [0.0, 1.0]
    assert tokenizer.N_var_per_element == 5
    assert mode in ["uniform", "gaussian", "negative"]

    # bbt = tokenizer.bbox_tokenizer
    V = 4
    N = tokenizer.N_bbox_per_var

    if tokenizer.is_loc_vocab_shared:
        slices = [slice(tokenizer.N_label, tokenizer.N_label + N) for i in range(V)]
    else:
        slices = [
            slice(tokenizer.N_label + i * N, tokenizer.N_label + (i + 1) * N)
            for i in range(V)
        ]

    logits = torch.zeros(
        (tokenizer.N_total, tokenizer.N_total),
    )
    logits.fill_diagonal_(1.0)

    for i, key in enumerate(GEO_KEYS):
        cluster_centers = tokenizer.bucketizers[key].centers.view(-1)
        ii, jj = torch.meshgrid(cluster_centers, cluster_centers, indexing="ij")
        if mode == "uniform":
            logits[slices[i], slices[i]] = (torch.abs(ii - jj) < offset_ratio).float()
        elif mode == "negative":
            logits[slices[i], slices[i]] = (torch.abs(ii - jj) >= offset_ratio).float()
        elif mode == "gaussian":
            # p(x) = a * exp( -(x-b)^2 / (2 * c^2))
            # -> log p(x) = log(a) - (x-b)^2 / (2 * c^2)
            # thus, a strength of adjustment is proportional to -(ii - jj)^2
            logits[slices[i], slices[i]] = -1.0 * (ii - jj) ** 2
        else:
            raise NotImplementedError

    logits = rearrange(F.embedding(seq, logits.to(seq.device)), "b s c -> b c s")
    return logits


def set_weak_logits_for_refinement(
    cond: ConditionalInputsForDiscreteLayout,
    tokenizer: LayoutSequenceTokenizer,
    sampling_cfg: DictConfig,
) -> ConditionalInputsForDiscreteLayout:
    """
    Set hand-crafted prior for the position/size of each element (Eq. 8)
    """
    assert cond.seq_observed is not None and cond.mask is not None
    w = sampling_cfg.refine_lambda
    if sampling_cfg.refine_mode == "negative":
        w *= -1.0

    cond.weak_mask = repeat(~cond.mask, "b s -> b c s", c=tokenizer.N_total)
    cond.weak_logits = _index_to_smoothed_log_onehot(
        cond.seq_observed,
        tokenizer,
        mode=sampling_cfg.refine_mode,
        offset_ratio=sampling_cfg.refine_offset_ratio,
    )
    cond.weak_logits *= w
    return cond


def _stochastic_convert(
    model_log_prob: Tensor,
    tokenizer: LayoutSequenceTokenizer,
) -> dict[str, Tensor]:
    """
    Convert model_log_prob (B, C, S) to average bbox location (B, S).
    """

    N = tokenizer.N_bbox_per_var
    step = tokenizer.N_var_per_element
    N_label = tokenizer.N_label
    outputs = {}
    for mult, key in enumerate(GEO_KEYS):
        attr_offset = tokenizer.var_order.index(key)
        if tokenizer.is_loc_vocab_shared:
            sl = slice(N_label, N_label + N)
        else:
            sl = slice(N_label + mult * N, N_label + (mult + 1) * N)
        bbox_logits = model_log_prob[:, sl, attr_offset::step]
        bbox_prob = F.softmax(bbox_logits, dim=1)

        centers = tokenizer.bucketizers[key].centers
        centers = rearrange(centers, "n s -> 1 n s").to(bbox_prob)
        outputs[key] = reduce(bbox_prob * centers, "e n s -> e s", reduction="sum")

    return outputs


@torch.set_grad_enabled(True)
def update_logits_for_relation(
    t: int,
    cond: ConditionalInputsForDiscreteLayout,
    model_log_prob: Tensor,  # (B, C, S)
    tokenizer: LayoutSequenceTokenizer,
    sampling_cfg: DictConfig = None,
) -> Tensor:
    """
    Update model_log_prob multiple times following Eq. 7.
    model_log_prob corresponds to p_{\theta}(\bm{z}_{t-1}|\bm{z}_{t}).
    """
    B = model_log_prob.size(0)
    # detach var. in order not to backpropagate thrhough diffusion model p_{\theta}.
    optim_target_log_prob = torch.nn.Parameter(model_log_prob.detach())

    # we found that adaptive optimizer does not work.
    optimizer = torch.optim.SGD(
        [optim_target_log_prob], lr=sampling_cfg.relation_lambda
    )

    device = model_log_prob.device
    T = 0 if t < 10 else sampling_cfg.relation_num_update
    for _ in range(T):
        optimizer.zero_grad()
        coords = _stochastic_convert(
            model_log_prob=optim_target_log_prob,
            tokenizer=tokenizer,
        )
        loss_batch = 0.0

        # batched implementation
        bbox_flatten = torch.stack(
            [coords[key] for key in GEO_KEYS], dim=-1
        )  # (B, S, 4)
        canvas = torch.tensor([0.5, 0.5, 1.0, 1.0])
        canvas = repeat(canvas, "x -> b 1 x", b=B)
        bbox_flatten = torch.cat(
            [
                canvas.to(device),
                bbox_flatten,
            ],
            dim=1,
        )  # (B, S + 1, 4)
        graph = Graph(
            edge_indexes=cond.edge_indexes, edge_attributes=cond.edge_attributes
        )
        loss = [f(bbox_flatten, graph) for f in relational_constraints]
        print(loss)
        loss_batch = torch.stack(loss).mean()

        loss_batch.backward()
        optimizer.step()

    return optim_target_log_prob.detach()
