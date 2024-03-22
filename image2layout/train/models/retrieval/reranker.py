import random
from typing import Any, Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from sklearn.metrics import pairwise_distances

np.random.seed(0)


def maximal_marginal_relevance(
    score_di_q: np.ndarray,
    score_di_dj: np.ndarray,
    lam: float,
    top_k: int,
    score_type: str,
) -> np.ndarray:
    """Computes the Maximal Marginal Relevance (Carbonell and Goldstein, 1998) given two similarity arrays.

    Score can be defined in terms of similarity or distance. Similarity is maximized, distance is minimized.

    Note:
        `.cpu()` is called to account for cases when the tensors are on GPU.
        If they are already in RAM, then `.cpu()` is a no-op.

    Args:
        score_di_q (T.Tensor): Scores between candidates and query, shape: (N,).
        score_di_dj (T.Tensor): Pairwise scores between candidates, shape: (N, N).
        lam (float): Lambda parameter: 0 = maximal diversity, 1 = greedy.
        num_iters (int): Number of results to return.
        score_type (str): One of ["similarity", "distance"].

    Returns:
        T.Tensor: Ordering of the candidates with respect to the MMR scoring.
    """
    _accepted_score_types = ["similarity", "distance"]
    if score_type not in _accepted_score_types:
        raise ValueError(f"score_type must be one of {_accepted_score_types}.")

    if not (0 < top_k <= len(score_di_q)):
        raise ValueError(f"Number of iterations must be in (0, {len(score_di_q)}].")

    if not (0 <= lam <= 1):
        raise ValueError("lambda must be in [0, 1].")

    R = np.arange(len(score_di_q))

    if score_type == "similarity":
        S = np.array([score_di_q.argmax()])
    elif score_type == "distance":
        S = np.array([score_di_q.argmin()])

    for _ in range(top_k - 1):
        cur_di = R[~np.isin(R, S)]  # Di in R\S

        lhs = score_di_q[cur_di]

        if score_type == "similarity":
            rhs = score_di_dj[rearrange(S, "s -> s 1"), cur_di].max(axis=0)
            idx = np.argmax(lam * lhs - (1 - lam) * rhs)
        elif score_type == "distance":
            rhs = score_di_dj[rearrange(S, "s -> s 1"), cur_di].min(axis=0)
            idx = np.argmin(lam * lhs - (1 - lam) * rhs)

        S = np.append(S, [cur_di[idx]])

    return S


def reranker_top_k(
    score_di_q: np.ndarray,
    top_k: int,
    score_type: str,
) -> np.ndarray:
    indexes = np.argsort(score_di_q)
    if score_type == "similarity":
        indexes = indexes[::-1]
    return indexes[:top_k]


def reranker_random(
    input_len: int,
    top_k: int,
) -> np.ndarray:
    return np.random.choice(input_len, top_k, replace=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # general setting
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feat_dim", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--reranker_name", type=str, default="mmr")

    # mmr
    parser.add_argument("--lam", type=float, default=1.0)
    args = parser.parse_args()

    all_indexes = list(range(args.batch_size))

    query = np.zeros((1, args.feat_dim))
    candidates = np.random.randn(args.batch_size, args.feat_dim)

    metric = "l1"
    score_di_q = pairwise_distances(query, candidates, metric=metric)[0]
    score_di_dj = pairwise_distances(candidates, candidates, metric=metric)

    def plot(
        ax: matplotlib.axes.Axes,
        candidates: np.ndarray,
        indexes: list[int],
        color: str,
        label: Optional[str] = None,
    ) -> None:
        assert candidates.ndim == 2 and candidates.shape[1] == 2
        ax.scatter(
            x=[candidates[i][0] for i in indexes],
            y=[candidates[i][1] for i in indexes],
            c=color,
        )
        if label:
            ax.set_title(label)
        ax.set_xlim([-3.0, 3.0])
        ax.set_ylim([-3.0, 3.0])

    mult = 4
    fig = plt.figure(figsize=(3 * mult, mult))

    # random
    ax = fig.add_subplot(1, 3, 1)
    indexes = random.choices(all_indexes, k=args.top_k)
    plot(ax=ax, candidates=candidates, indexes=all_indexes, color="#eeeeee")
    plot(ax=ax, candidates=candidates, indexes=indexes, label="random", color="#555555")
    plot(
        ax=ax,
        candidates=query,
        indexes=[
            0,
        ],
        color="#ff0000",
    )

    # topk
    ax = fig.add_subplot(1, 3, 2)
    indexes = np.argsort(score_di_q)[: args.top_k]
    plot(ax=ax, candidates=candidates, indexes=all_indexes, color="#eeeeee")
    plot(ax=ax, candidates=candidates, indexes=indexes, label="topk", color="#555555")
    plot(
        ax=ax,
        candidates=query,
        indexes=[
            0,
        ],
        color="#ff0000",
    )

    # mmr
    ax = fig.add_subplot(1, 3, 3)

    indexes = maximal_marginal_relevance(
        score_di_q,
        score_di_dj,
        score_type="distance",
        top_k=args.top_k,
        lam=args.lam,
    )
    plot(ax=ax, candidates=candidates, indexes=all_indexes, color="#eeeeee")
    plot(ax=ax, candidates=candidates, indexes=indexes, label="mmr", color="#555555")
    plot(
        ax=ax,
        candidates=query,
        indexes=[
            0,
        ],
        color="#ff0000",
    )

    plt.savefig("comparison_retrieval.png")
