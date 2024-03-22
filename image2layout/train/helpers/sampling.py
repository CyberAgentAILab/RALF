import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor

FILTER_VALUE = -float("Inf")


def top_k_logits(logits: Tensor, k: int, dim: int = -1) -> Tensor:
    # logits: (B, C)
    v, _ = torch.topk(logits, k, dim)
    out = logits.clone()
    out[out < v[:, [-1]]] = FILTER_VALUE
    return out


def sample(logits: Tensor, sampling_cfg: DictConfig, temperature=None) -> Tensor:
    """
    Input: logits (B, C, *N)
    Output: (B, 1, *N)
    """
    assert logits.ndim in [2, 3]
    if sampling_cfg.name == "deterministic":
        output = torch.argmax(logits, dim=1, keepdim=True)
    else:

        if temperature is None:
            temperature = sampling_cfg.temperature

        logits_ = logits / temperature

        if sampling_cfg.name == "top_k":
            logits = top_k_logits(logits_, k=sampling_cfg.top_k, dim=1)
        elif sampling_cfg.name == "top_p":
            top_p = sampling_cfg.top_p
            assert 0.0 < top_p <= 1.0

            S = logits.size(1)
            # https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order-after-torch-sort
            sorted_logits, sorted_indices = torch.sort(logits_, descending=True, dim=1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)

            indices = torch.arange(S).view(1, S).to(logits.device)
            if logits.ndim == 3:
                indices = indices.unsqueeze(dim=-1)

            # make sure to keep the first logit (most likely one)
            sorted_logits[(cumulative_probs > top_p) & (indices > 0)] = FILTER_VALUE
            logits = sorted_logits.gather(dim=1, index=sorted_indices.argsort(dim=1))
        elif sampling_cfg.name == "random":
            logits = logits_
        elif sampling_cfg.name == "gumbel":
            uniform = torch.rand_like(logits_)
            const = 1e-30
            gumbel_noise = -torch.log(-torch.log(uniform + const) + const)
            logits = logits_ + gumbel_noise
        else:
            raise NotImplementedError

        probs = F.softmax(logits, dim=1)
        if probs.ndim == 2:
            output = torch.multinomial(probs, num_samples=1)  # (B, 1)
        elif probs.ndim == 3:
            S = probs.shape[2]
            probs = rearrange(probs, "b c s -> (b s) c")
            output = torch.multinomial(probs, num_samples=1)
            output = rearrange(output, "(b s) 1 -> b 1 s", s=S)
        else:
            raise NotImplementedError
    return output
