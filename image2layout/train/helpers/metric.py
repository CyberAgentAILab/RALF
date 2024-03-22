import itertools
import logging
import multiprocessing
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Union

import cv2
import datasets as ds
import numpy as np
import timm
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from prdc import compute_prdc
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import Tensor
from torchvision import transforms

from .util import convert_xywh_to_ltrb, is_list_of_dict, list_of_dict_to_dict_of_list

logger = logging.getLogger(__name__)

Layout = tuple[np.ndarray, np.ndarray]
IS_DEBUG = True


def _mean(values: list[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    else:
        return sum(values) / len(values)


def compute_generative_model_scores(
    feats_real: Union[Tensor, np.ndarray],
    feats_fake: Union[Tensor, np.ndarray],
) -> dict[str, float]:
    """
    Compute precision, recall, density, coverage, and FID.
    """
    if torch.is_tensor(feats_real):  # type: ignore
        feats_real = feats_real.numpy()  # type: ignore
    if torch.is_tensor(feats_fake):  # type: ignore
        feats_fake = feats_fake.numpy()  # type: ignore

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    results = compute_prdc(
        real_features=feats_real, fake_features=feats_fake, nearest_k=5
    )
    results["fid"] = calculate_frechet_distance(
        mu_real, sigma_real, mu_fake, sigma_fake
    )
    return {k: float(v) for (k, v) in results.items()}


def _get_coords(
    batch: dict[str, Tensor], validate_range: bool = True
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    xc, yc = batch["center_x"], batch["center_y"]
    xl = xc - batch["width"] / 2.0
    xr = xc + batch["width"] / 2.0
    yt = yc - batch["height"] / 2.0
    yb = yc + batch["height"] / 2.0

    if validate_range:
        xl = torch.maximum(xl, torch.zeros_like(xl))
        xr = torch.minimum(xr, torch.ones_like(xr))
        yt = torch.maximum(yt, torch.zeros_like(yt))
        yb = torch.minimum(yb, torch.ones_like(yb))
    return xl, xc, xr, yt, yc, yb


def compute_alignment(batch: dict[str, Tensor]) -> dict[str, list[float]]:
    """
    Computes some alignment metrics that are different to each other in previous works.
    Lower values are generally better.
    Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    """
    xl, xc, xr, yt, yc, yb = _get_coords(batch)
    mask = batch["mask"]
    _, S = mask.size()

    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log10(1 - X)

    # original
    # return X.sum(-1) / mask.float().sum(-1)

    score = reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    Y = torch.stack([xl, xc, xr], dim=1)
    Y = rearrange(Y, "b x s -> b x 1 s") - rearrange(Y, "b x s -> b x s 1")

    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=Y.device)
    batch_mask[:, idx, idx] = True
    batch_mask = repeat(batch_mask, "b s1 s2 -> b x s1 s2", x=3)
    Y[batch_mask] = 1.0

    # Y = rearrange(Y.abs(), "b x s1 s2 -> b s1 x s2")
    # Y = reduce(Y, "b x s1 s2 -> b x", "min")
    # Y = rearrange(Y.abs(), " -> b s1 x s2")
    Y = reduce(Y.abs(), "b x s1 s2 -> b s1", "min")
    Y[Y == 1.0] = 0.0
    score_Y = reduce(Y, "b s -> b", "sum")

    results = {
        # "alignment-ACLayoutGAN": score,  # Because it may be confusing.
        "alignment-LayoutGAN++": score_normalized,
        # "alignment-NDN": score_Y,  # Because it may be confusing.
    }
    return {k: v.tolist() for (k, v) in results.items()}


def compute_overlap(batch: dict[str, Tensor]) -> dict[str, list[float]]:
    """
    Based on
    (i) Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    (ii) LAYOUTGAN: GENERATING GRAPHIC LAYOUTS WITH WIREFRAME DISCRIMINATORS (ICLR2019)
    https://arxiv.org/abs/1901.06767
    "percentage of total overlapping area among any two bounding boxes inside the whole page."
    Lower values are generally better.
    At least BLT authors seems to sum. (in the MSCOCO case, it surpasses 1.0)
    """
    mask = batch["mask"]
    B, S = mask.size()
    for key in ["center_x", "center_y", "width", "height"]:
        batch[key][~mask] = 0.0

    l1, _, r1, t1, _, b1 = [x.unsqueeze(dim=-1) for x in _get_coords(batch)]
    l2, _, r2, t2, _, b2 = [x.unsqueeze(dim=-2) for x in _get_coords(batch)]

    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    # diag_mask = torch.eye(a1.size(1), dtype=torch.bool, device=a1.device)
    # ai = ai.masked_fill(diag_mask, 0)
    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=ai.device)
    batch_mask[:, idx, idx] = True
    ai = ai.masked_fill(batch_mask, 0)

    ar = torch.nan_to_num(ai / a1)  # (B, S, S)

    # original
    # return ar.sum(dim=(1, 2)) / mask.float().sum(-1)

    # fixed to avoid the case with single bbox
    score = reduce(ar, "b s1 s2 -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    ids = torch.arange(S)
    ii, jj = torch.meshgrid(ids, ids, indexing="ij")
    ai[repeat(ii >= jj, "s1 s2 -> b s1 s2", b=B)] = 0.0
    overlap = reduce(ai, "b s1 s2 -> b", reduction="sum")

    results = {
        # "overlap-ACLayoutGAN": score,  # Because it may be confusing.
        "overlap-LayoutGAN++": score_normalized,
        # "overlap-LayoutGAN": overlap,  # Because it may be confusing.
    }
    return {k: v.tolist() for (k, v) in results.items()}


def _compute_iou(
    box_1: Union[np.ndarray, Tensor],
    box_2: Union[np.ndarray, Tensor],
    method: str = "iou",
) -> np.ndarray:
    """
    Since there are many IoU-like metrics,
    we compute them at once and return the specified one.
    box_1 and box_2 are in (N, 4) format.
    """
    assert method in ["iou", "giou", "ai/a1", "ai/a2"]

    if isinstance(box_1, Tensor):
        box_1 = np.array(box_1)
        box_2 = np.array(box_2)
    assert len(box_1) == len(box_2)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if method == "iou":
        return iou
    elif method == "ai/a1":
        return ai / a1
    elif method == "ai/a2":
        return ai / a2

    # outer region
    l_min = np.minimum(l1, l2)
    r_max = np.maximum(r1, r2)
    t_min = np.minimum(t1, t2)
    b_max = np.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou: np.ndarray = iou - (ac - au) / ac

    return giou


def _compute_perceptual_iou(
    box_1: Union[np.ndarray, Tensor],
    box_2: Union[np.ndarray, Tensor],
) -> np.ndarray:
    """
    Computes 'Perceptual' IoU used in
    BLT: Bidirectional Layout Transformer for Controllable Layout Generation [Kong+, BLT'22]
    https://arxiv.org/abs/2112.05112
    box_1 and box_2 are in (N, 4) format.
    """
    if isinstance(box_1, Tensor):
        box_1 = np.array(box_1)
        box_2 = np.array(box_2)
    assert len(box_1) == len(box_2)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1 = (r1 - l1) * (b1 - t1)
    # a2 = (r2 - l2) * (b2 - t2)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))

    unique_box_1 = np.unique(box_1, axis=0)
    N = 32
    l1, t1, r1, b1 = [
        (x * N).round().astype(np.int32).clip(0, N)
        for x in convert_xywh_to_ltrb(unique_box_1.T)
    ]
    canvas = np.zeros((N, N))
    for (l, t, r, b) in zip(l1, t1, r1, b1):
        canvas[t:b, l:r] = 1
    global_area_union = canvas.sum() / (N**2)

    if global_area_union > 0.0:
        iou: np.ndarray = ai / global_area_union
        return iou
    else:
        return np.zeros((1,))


IOU_FUNC_FACTORY: dict[str, Callable] = {
    "iou": partial(_compute_iou, method="iou"),
    "ai/a1": partial(_compute_iou, method="ai/a1"),
    "ai/a2": partial(_compute_iou, method="ai/a2"),
    "giou": partial(_compute_iou, method="giou"),
    "perceptual": _compute_perceptual_iou,
}


def iou_func_factory(name: str = "iou") -> Callable:
    return IOU_FUNC_FACTORY[name]


def _list_all_pair_indices(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all pairs
    """
    N = bbox.shape[0]
    ii, jj = np.meshgrid(range(N), range(N))
    ii, jj = ii.flatten(), jj.flatten()
    is_non_diag = ii != jj  # IoU for diag is always 1.0
    ii, jj = ii[is_non_diag], jj[is_non_diag]
    return ii, jj


def run_parallel(  # type: ignore
    func: Callable,
    layouts: list[Layout],
    is_debug: bool = IS_DEBUG,
    n_jobs: Optional[int] = None,
):
    """
    Assumption:
    each func returns a single value or dict where each element is a single value
    """
    if is_debug:
        scores = [func(layout) for layout in layouts]
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(func, layouts)

    if is_list_of_dict(scores):
        filtered_scores = list_of_dict_to_dict_of_list(scores)
        for k in filtered_scores:
            filtered_scores[k] = [s for s in filtered_scores[k] if s is not None]
        return filtered_scores
    else:
        return [s for s in scores if s is not None]


# Below are from:
# PosterLayout: A New Benchmark and Approach for
# Content-aware Visual-Textual Presentation Layout (CVPR2023)
# https://arxiv.org/abs/2303.15937


def compute_validity(
    data: list[dict[str, Tensor]],
    thresh: float = 1e-3,
) -> tuple[list[dict[str, Tensor]], float]:
    """
    Ratio of valid elements to all elements in the layout used in PosterLayout,
    where the area must be greater than 0.1% of the canvas.
    For validity, higher values are better (in 0.0 - 1.0 range).
    """
    filtered_data = []
    N_numerator, N_denominator = 0, 0
    for d in data:
        is_valid = [(w * h > thresh) for (w, h) in zip(d["width"], d["height"])]
        N_denominator += len(is_valid)
        N_numerator += is_valid.count(True)

        filtered_d = {}
        for key, value in d.items():
            if isinstance(value, list):
                filtered_d[key] = []
                assert len(value) == len(
                    is_valid
                ), f"{len(value)} != {len(is_valid)}, value: {value}, is_valid: {is_valid}"
                for j in range(len(is_valid)):
                    if is_valid[j]:
                        filtered_d[key].append(value[j])
            else:
                filtered_d[key] = value
        filtered_data.append(filtered_d)

    validity = N_numerator / N_denominator
    return filtered_data, validity


def __compute_overlay(layout: Layout) -> Optional[float]:
    """
    Average IoU except underlay components used in PosterLayout.
    Lower values are better (in 0.0 - 1.0 range).
    """
    bbox, _ = layout
    N = bbox.shape[0]
    if N in [0, 1]:
        return None  # no overlap in principle

    ii, jj = _list_all_pair_indices(bbox)
    iou: np.ndarray = iou_func_factory("iou")(bbox[ii], bbox[jj])
    result: float = iou.mean().item()
    return result


def compute_overlay(
    batch: dict[str, Tensor],
    feature_label: ds.ClassLabel,
) -> dict[str, list[float]]:
    """
    See __compute_overlay for detailed description.
    """
    underlay_id = feature_label.str2int("underlay")

    layouts = []
    for i in range(batch["label"].size(0)):
        new_mask = batch["mask"][i] & (
            batch["label"][i] != underlay_id
        )  # ignore underlay
        label = batch["label"][i][new_mask]
        bbox = []
        for key in ["center_x", "center_y", "width", "height"]:
            bbox.append(batch[key][i][new_mask])
        bbox = torch.stack(bbox, dim=-1)  # type: ignore
        layouts.append((np.array(bbox), np.array(label)))

    results: dict[str, list[float]] = {
        "overlay": run_parallel(__compute_overlay, layouts)
    }
    return results


def __compute_underlay_effectiveness(
    layout: Layout,
    underlay_id: int,
) -> dict[str, Optional[float]]:
    """
    Ratio of valid underlay elements to total underlay elements used in PosterLayout.
    Intuitively, underlay should be placed under other non-underlay elements.
    - strict: scoring the underlay as
        1: there is a non-underlay element completely inside
        0: otherwise
    - loose: Calcurate (ai/a2).
    Aggregation part is following the original code (description in paper is not enough).
    Higher values are better (in 0.0 - 1.0 range).
    """
    bbox, label = layout
    N = bbox.shape[0]
    if N in [0, 1]:
        # no overlap in principle
        return {
            "underlay_effectiveness_loose": None,
            "underlay_effectiveness_strict": None,
        }

    ii, jj = _list_all_pair_indices(bbox)
    iou = iou_func_factory("ai/a2")(bbox[ii], bbox[jj])
    mat, mask = np.zeros((N, N)), np.full((N, N), fill_value=False)
    mat[ii, jj] = iou
    mask[ii, jj] = True

    # mask out iou between underlays
    underlay_inds = [i for (i, id_) in enumerate(label) if id_ == underlay_id]
    for (i, j) in itertools.product(underlay_inds, underlay_inds):
        mask[i, j] = False

    loose_scores, strict_scores = [], []
    for i in range(N):
        if label[i] != underlay_id:
            continue

        score = mat[i][mask[i]]
        if len(score) > 0:
            loose_score = score.max()

            # if ai / a2 is (almost) 1.0, it means a2 is completely inside a1
            # if we can find any non-underlay object inside the underlay, it is ok
            # thresh is used to avoid numerical small difference
            thresh = 1.0 - np.finfo(np.float32).eps
            strict_score = (score >= thresh).any().astype(np.float32)
        else:
            loose_score = 0.0
            strict_score = 0.0
        loose_scores.append(loose_score)
        strict_scores.append(strict_score)

    return {
        "underlay_effectiveness_loose": _mean(loose_scores),
        "underlay_effectiveness_strict": _mean(strict_scores),
    }


def compute_underlay_effectiveness(
    batch: dict[str, Tensor],
    feature_label: ds.ClassLabel,
) -> dict[str, list[float]]:
    """
    See __compute_underlay_effectiveness for detailed description.
    """
    underlay_id = feature_label.str2int("underlay")

    layouts = []
    for i in range(batch["label"].size(0)):
        mask = batch["mask"][i]
        label = batch["label"][i][mask]
        bbox = []
        for key in ["center_x", "center_y", "width", "height"]:
            bbox.append(batch[key][i][mask])
        bbox = torch.stack(bbox, dim=-1)  # type: ignore
        layouts.append((np.array(bbox), np.array(label)))

    results: dict[str, list[float]] = run_parallel(
        partial(__compute_underlay_effectiveness, underlay_id=underlay_id), layouts
    )
    return results


def _extract_grad(image: Tensor) -> Tensor:
    image_npy = rearrange(np.array(image * 255), "c h w -> h w c")
    image_npy_gray = cv2.cvtColor(image_npy, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(image_npy_gray, -1, 1, 0)
    grad_y = cv2.Sobel(image_npy_gray, -1, 0, 1)
    grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
    # ?: is it really OK to do content adaptive normalization?
    grad_xy = grad_xy / np.max(grad_xy)
    return torch.from_numpy(grad_xy)


def compute_saliency_aware_metrics(
    batch: dict[str, Tensor],
    feature_label: ds.ClassLabel,
) -> dict[str, list[float]]:
    """
    - utilization:
        Utilization rate of space suitable for arranging elements,
        Higher values are generally better (in 0.0 - 1.0 range).
    - occlusion:
        Average saliency of areas covered by elements.
        Lower values are generally better (in 0.0 - 1.0 range).
    - unreadability:
        Non-flatness of regions that text elements are solely put on
        Lower values are generally better.
    """
    text_id = feature_label.str2int("text")
    underlay_id = feature_label.str2int("underlay")

    B, _, H, W = batch["saliency"].size()
    saliency = rearrange(batch["saliency"], "b 1 h w -> b h w")
    inv_saliency = 1.0 - saliency
    xl, _, xr, yt, _, yb = _get_coords(batch)

    results = defaultdict(list)
    for i in range(B):
        mask = batch["mask"][i]
        left = (xl[i][mask] * W).round().int().tolist()
        top = (yt[i][mask] * H).round().int().tolist()
        right = (xr[i][mask] * W).round().int().tolist()
        bottom = (yb[i][mask] * H).round().int().tolist()

        bbox_mask = torch.zeros((H, W))
        for (l, t, r, b) in zip(left, top, right, bottom):
            bbox_mask[t:b, l:r] = 1

        # utilization
        numerator = torch.sum(inv_saliency[i] * bbox_mask)
        denominator = torch.sum(inv_saliency[i])
        assert denominator > 0.0
        results["utilization"].append((numerator / denominator).item())

        # occlusion
        occlusion = saliency[i][bbox_mask.bool()]
        if len(occlusion) == 0:
            results["occlusion"].append(0.0)
        else:
            results["occlusion"].append(occlusion.mean().item())

        # unreadability
        # note: values are much smaller than repoted probably because
        # they compute gradient in 750*513
        bbox_mask_special = torch.zeros((H, W))
        label = batch["label"][i].tolist()

        for (id_, l, t, r, b) in zip(label, left, top, right, bottom):
            # get text area
            if id_ == text_id:
                bbox_mask_special[t:b, l:r] = 1
        for (id_, l, t, r, b) in zip(label, left, top, right, bottom):
            # subtract underlay area
            if id_ == underlay_id:
                bbox_mask_special[t:b, l:r] = 0

        g_xy = _extract_grad(batch["image"][i])
        unreadability = g_xy[bbox_mask_special.bool()]
        if len(unreadability) == 0:
            results["unreadability"].append(0.0)
        else:
            results["unreadability"].append(unreadability.mean().item())

    return results


# Below are from:
# Composition-aware Graphic Layout GAN for Visual-textual Presentation Designs (IJCAI2022)
# https://arxiv.org/abs/2205.00303


def compute_sub(
    batch: dict[str, Tensor],
    feature_label: ds.ClassLabel,
) -> dict[str, list[float]]:
    """
    Judge whether the promoted product is gaining much attention (denoted as R_{sub}).
    To compute, one should get attention maps of the promoted product
    (queried by their category tags extracted on product pages) by CLIP.
    Not currently implemented because the publicly available dataset
    does not contain such information.
    """
    raise NotImplementedError


class _TimmVGGWrapper(nn.Module):
    """
    Wrapper class to adjust singleton pattern of SingletonVGG.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone_tag = "hf_hub:timm/vgg16.tv_in1k"
        logger.info(f"Loading timm model from {backbone_tag=}")
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(backbone_tag, pretrained=True, num_classes=0).to(
            torch.device(device_name)
        )
        self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        # transform = timm.data.create_transform(**data_config, is_training=False)
        # transform = [
        #     t for t in transform.transforms if not isinstance(t, transforms.ToTensor)
        # ]
        # self.transform = transforms.Compose(transform)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), transforms.InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
            ]
        )
        logger.info(f"Transform of VGG: {self.transform}")

    def forward(self, images: Tensor) -> Tensor:
        assert images.ndim == 4
        h = torch.stack([self.transform(image) for image in images])
        h = self.model(h)
        return h


class _TimmInceptionV3Wrapper(nn.Module):
    """
    Wrapper class to adjust singleton pattern of SingletonInceptionV3.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info(f"Loading timm model from inception_v3")
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(
            "inception_v3", pretrained=True, num_classes=0
        ).to(torch.device(device_name))
        self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        # transform = timm.data.create_transform(**data_config, is_training=False)
        # transform = [
        #     t for t in transform.transforms if not isinstance(t, transforms.ToTensor)
        # ]
        # self.transform = transforms.Compose(transform)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (299, 299), transforms.InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
            ]
        )
        logger.info(f"Transform of InceptionV3: {self.transform}")

    def forward(self, images: Tensor) -> Tensor:
        assert images.size(1) == 3
        assert 0.0 <= images.min().item() and images.max().item() <= 1.0
        h = torch.stack([self.transform(image) for image in images])
        h = self.model(h)
        return h


class SingletonTimmVGG(object):
    """
    Follow singleton pattern to avoid loading VGG16 multiple times.
    """

    def __new__(cls):  # type: ignore
        if not hasattr(cls, "instance"):
            cls.instance = _TimmVGGWrapper()
        return cls.instance  # type: ignore


class SingletonTimmInceptionV3(object):
    """
    Follow singleton pattern to avoid loading InceptionV3 multiple times.
    """

    def __new__(cls):  # type: ignore
        if not hasattr(cls, "instance"):
            cls.instance = _TimmInceptionV3Wrapper()
        return cls.instance  # type: ignore


def compute_rshm(
    batch: dict[str, Tensor],
) -> dict[str, list[float]]:
    """
    Measure the occlusion levels of key subjects (denoted as R_{shm}).
    We feed the salient images with or without layout regions masked
    into a pretrained VGG16, and calculate L2 distance between their output logits.
    Lower values are generally better (in 0.0 - 1.0 range).
    """
    # unlike compute_saliency_aware_metrics, do batch processing as much as possible
    vgg16 = SingletonTimmVGG()  # type: ignore
    images = batch["image"].clone()
    B, _, H, W = images.size()

    # get layout masks
    xl, _, xr, yt, _, yb = _get_coords(batch)
    layout_masks: list[Tensor] = []
    for i in range(B):
        mask = batch["mask"][i]
        left = (xl[i][mask] * W).round().int().tolist()
        top = (yt[i][mask] * H).round().int().tolist()
        right = (xr[i][mask] * W).round().int().tolist()
        bottom = (yb[i][mask] * H).round().int().tolist()

        bbox_mask = torch.full((1, H, W), fill_value=False)
        for (l, t, r, b) in zip(left, top, right, bottom):
            bbox_mask[:, t:b, l:r] = True
        layout_masks.append(bbox_mask)
    layout_masks = torch.stack(layout_masks)  # type: ignore
    layout_masks = repeat(layout_masks, "b 1 h w -> b c h w", c=3)

    images_masked = batch["image"].clone()
    images_masked[layout_masks] = 0.5  # mask by gray values, is it ok?
    with torch.no_grad():
        logits = vgg16(images.cuda()).detach().cpu()  # type: ignore
        logits_masked = vgg16(images_masked.cuda()).detach().cpu()  # type: ignore

    dist = torch.linalg.vector_norm(
        logits_masked - logits, dim=1
    )  # L2 dist from (B, 1000)
    return {"R_{shm} (vgg distance)": dist.tolist()}
