from functools import partial
from typing import Any, Callable, NamedTuple

import torch
from image2layout.train.helpers.relationships import REL_SIZE_ALPHA, RelLoc, RelSize
from image2layout.train.helpers.util import convert_xywh_to_ltrb


class Graph(NamedTuple):
    edge_indexes: torch.Tensor  # (E, 2), usually >=0, <0 means invalid edge
    edge_attributes: torch.Tensor  # (E, )


def less_equal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(a - b)


def less(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.relu(a - b + eps)


def safe_gather(input_: torch.Tensor, index: torch.LongTensor) -> torch.Tensor:
    """
    torch.gather raises an error when index is negative.
    negative indexes are replaced with zero, but make sure not to use it.
    input_: (B, S)
    index: (B, E)
    """
    index = index.masked_fill(index < 0, 0)
    return torch.gather(input_, dim=1, index=index)


def _relation_size(
    rel_value: RelSize,
    cost_func: Callable,
    bbox_flatten: torch.Tensor,
    graph: Graph,
    canvas: bool,
) -> torch.Tensor:
    assert bbox_flatten.ndim == 3
    cond = graph.edge_indexes[..., 0].eq(0).eq(canvas)
    cond &= (graph.edge_attributes & 1 << rel_value).ne(0)
    a = bbox_flatten[:, :, 2] * bbox_flatten[:, :, 3]

    ai = safe_gather(a, graph.edge_indexes[..., 0])
    aj = safe_gather(a, graph.edge_indexes[..., 1])
    cost = cost_func(ai, aj).masked_fill(~cond, 0)
    return cost.sum(dim=1).mean()


def relation_size_sm(
    bbox_flatten: torch.Tensor, graph: Graph, canvas: bool = False
) -> torch.Tensor:
    def cost_func(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        # a2 <= a1_sm
        a1_sm = (1 - REL_SIZE_ALPHA) * a1
        return less_equal(a2, a1_sm)

    return _relation_size(RelSize.SMALLER, cost_func, bbox_flatten, graph, canvas)


def relation_size_eq(
    bbox_flatten: torch.Tensor, graph: Graph, canvas: bool = False
) -> torch.Tensor:
    def cost_func(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        # a1_sm < a2 and a2 < a1_lg
        a1_sm = (1 - REL_SIZE_ALPHA) * a1
        a1_lg = (1 + REL_SIZE_ALPHA) * a1
        return less(a1_sm, a2) + less(a2, a1_lg)

    return _relation_size(RelSize.EQUAL, cost_func, bbox_flatten, graph, canvas)


def relation_size_lg(
    bbox_flatten: torch.Tensor, graph: Graph, canvas: bool = False
) -> torch.Tensor:
    def cost_func(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        # a1_lg <= a2
        a1_lg = (1 + REL_SIZE_ALPHA) * a1
        return less_equal(a1_lg, a2)

    return _relation_size(RelSize.LARGER, cost_func, bbox_flatten, graph, canvas)


def _relation_loc_canvas(
    rel_value: RelLoc, cost_func: Callable, bbox_flatten: torch.Tensor, graph: Graph
) -> torch.Tensor:
    assert bbox_flatten.ndim == 3
    cond = graph.edge_indexes[..., 0].eq(0)  # select canvas element as i-th
    cond &= (graph.edge_attributes & 1 << rel_value).ne(0)

    yc = bbox_flatten[:, :, 1]
    yc = safe_gather(yc, graph.edge_indexes[..., 1])
    cost = cost_func(yc).masked_fill(~cond, 0)
    return cost.sum(dim=1).mean()


def relation_loc_canvas_t(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(yc: torch.Tensor) -> Callable:
        # yc <= y_sm
        y_sm = 1.0 / 3
        return less_equal(yc, y_sm)

    return _relation_loc_canvas(RelLoc.TOP, cost_func, bbox_flatten, graph)


def relation_loc_canvas_c(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(yc: torch.Tensor) -> Callable:
        # y_sm < yc and yc < y_lg
        y_sm, y_lg = 1.0 / 3, 2.0 / 3
        return less(y_sm, yc) + less(yc, y_lg)

    return _relation_loc_canvas(RelLoc.CENTER, cost_func, bbox_flatten, graph)


def relation_loc_canvas_b(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(yc: torch.Tensor) -> Callable:
        # y_lg <= yc
        y_lg = 2.0 / 3
        return less_equal(y_lg, yc)

    return _relation_loc_canvas(RelLoc.BOTTOM, cost_func, bbox_flatten, graph)


def _relation_loc(
    rel_value: RelLoc, cost_func: Callable, bbox_flatten: torch.Tensor, graph: Graph
) -> torch.Tensor:
    assert bbox_flatten.ndim == 3
    cond = graph.edge_indexes[..., 0] > 0  # select non-canvas element as i-th
    cond &= (graph.edge_attributes & 1 << rel_value).ne(0)
    l, t, r, b = convert_xywh_to_ltrb(bbox_flatten.permute(2, 0, 1))

    z = graph.edge_indexes
    li, lj = safe_gather(l, z[..., 0]), safe_gather(l, z[..., 1])
    ti, tj = safe_gather(t, z[..., 0]), safe_gather(t, z[..., 1])
    ri, rj = safe_gather(r, z[..., 0]), safe_gather(r, z[..., 1])
    bi, bj = safe_gather(b, z[..., 0]), safe_gather(b, z[..., 1])

    cost = cost_func(l1=li, t1=ti, r1=ri, b1=bi, l2=lj, t2=tj, r2=rj, b2=bj)

    if rel_value in [RelLoc.LEFT, RelLoc.RIGHT, RelLoc.CENTER]:
        # t1 < b2 and t2 < b1
        cost = cost + less(ti, bj) + less(tj, bi)

    cost = cost.masked_fill(~cond, 0)
    return cost.sum(dim=1).mean()


def relation_loc_t(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(b2: torch.Tensor, t1: torch.Tensor, **kwargs: Any) -> Callable:
        # b2 <= t1
        return less_equal(b2, t1)

    return _relation_loc(RelLoc.TOP, cost_func, bbox_flatten, graph)


def relation_loc_b(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(b1: torch.Tensor, t2: torch.Tensor, **kwargs: Any) -> Callable:
        # b1 <= t2
        return less_equal(b1, t2)

    return _relation_loc(RelLoc.BOTTOM, cost_func, bbox_flatten, graph)


def relation_loc_l(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(r2: torch.Tensor, l1: torch.Tensor, **kwargs: Any) -> Callable:
        # r2 <= l1
        return less_equal(r2, l1)

    return _relation_loc(RelLoc.LEFT, cost_func, bbox_flatten, graph)


def relation_loc_r(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(r1: torch.Tensor, l2: torch.Tensor, **kwargs: Any) -> Callable:
        # r1 <= l2
        return less_equal(r1, l2)

    return _relation_loc(RelLoc.RIGHT, cost_func, bbox_flatten, graph)


def relation_loc_c(bbox_flatten: torch.Tensor, graph: Graph) -> torch.Tensor:
    def cost_func(
        l1: torch.Tensor,
        r2: torch.Tensor,
        l2: torch.Tensor,
        r1: torch.Tensor,
        **kwargs: Any,
    ) -> Callable:
        # l1 < r2 and l2 < r1
        return less(l1, r2) + less(l2, r1)

    return _relation_loc(RelLoc.CENTER, cost_func, bbox_flatten, graph)


relation = [
    partial(relation_size_sm, canvas=False),
    partial(relation_size_sm, canvas=True),
    partial(relation_size_eq, canvas=False),
    partial(relation_size_eq, canvas=True),
    partial(relation_size_lg, canvas=False),
    partial(relation_size_lg, canvas=True),
    relation_loc_canvas_t,
    relation_loc_canvas_c,
    relation_loc_canvas_b,
    relation_loc_t,
    relation_loc_b,
    relation_loc_l,
    relation_loc_r,
    relation_loc_c,
]
