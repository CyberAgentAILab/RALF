import logging
import pickle
from typing import Optional, Type

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import KMeans
from torch import LongTensor, Tensor

logger = logging.getLogger(__name__)


def get_kmeans_cluster_center(
    key: str,
    weight_path: Optional[str] = None,
    weights: Optional[dict[str, KMeans]] = None,
) -> Tensor:
    """
    Given a pre-loaded weights or path to the weights,
    return the k-means cluster centers
    """
    assert weight_path is not None or weights is not None

    if weight_path:
        fs, path_prefix = fsspec.core.url_to_fs(weight_path)
        logger.info(f"Load {weight_path=}")
        with fs.open(path_prefix, "rb") as f:
            weights = pickle.load(f)

    cluster_centers = np.sort(weights[key].cluster_centers_, axis=0)  # (N, 1)
    cluster_centers = cluster_centers[:, 0]
    return torch.from_numpy(cluster_centers)


class BaseBucketizer:
    """
    Interface for bucketizer to convert continuous / discrete variables
    Subclasses should override self._boundaries and self._centers
    """

    def __init__(self, n_boundaries: int = 128) -> None:
        self._n_boundaries = n_boundaries

        # below should be overriden
        self._boundaries = torch.tensor([])
        self._centers = torch.tensor([])

    def __call__(self, data: Tensor) -> Tensor:
        data = torch.clamp(data, min=0.0, max=1.0)
        return torch.bucketize(data, self._boundaries)

    def encode(self, data: Tensor) -> LongTensor:
        return self(data)  # type: ignore

    def decode(self, index: LongTensor) -> Tensor:
        index = torch.clamp(index, min=0, max=len(self._centers) - 1)  # type: ignore
        return F.embedding(index, self._centers)[..., 0]

    @property
    def boundaries(self) -> Tensor:
        return self._boundaries

    @property
    def centers(self) -> Tensor:
        return self._centers


class _LinearBucketizer(BaseBucketizer):
    """
    Uniform bucketization between 0.0 to 1.0
    """

    def __init__(self, n_boundaries: int = 128) -> None:
        super().__init__(n_boundaries)
        arr = torch.arange(self._n_boundaries + 1) / self._n_boundaries
        starts, ends = arr[:-1], arr[1:]
        self._boundaries = ends
        self._centers = rearrange((starts + ends) / 2.0, "n -> n 1")


class _KMeansBucketizer(BaseBucketizer):
    """
    Adaptive bucketization based on pre-computed features
    """

    def __init__(
        self,
        cluster_centers: Tensor,
        n_boundaries: int = 128,
    ) -> None:
        super().__init__(n_boundaries)

        ends = (cluster_centers[:-1] + cluster_centers[1:]) / 2.0
        ends = torch.cat([ends, torch.ones((1,))])

        # TODO: check if it's really OK (centers is not in center)
        self._centers = rearrange(cluster_centers, "n -> n 1")
        self._boundaries = ends


_BUCKETIZER_FACTORY = {
    "linear": _LinearBucketizer,
    "kmeans": _KMeansBucketizer,
}


def bucketizer_factory(name: str) -> Type:
    assert name in _BUCKETIZER_FACTORY, name
    return _BUCKETIZER_FACTORY[name]
