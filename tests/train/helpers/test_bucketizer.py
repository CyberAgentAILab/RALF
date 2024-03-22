import os
import random
from typing import Tuple

import torch
from image2layout.train.config.dataset import dataset_config_names
from image2layout.train.global_variables import GEO_KEYS, PRECOMPUTED_WEIGHT_DIR
from image2layout.train.helpers.bucketizer import (
    bucketizer_factory,
    get_kmeans_cluster_center,
)
from tests.util import repeat_func

DATASET_NAMES = ["cgl", "pku10"]


def _make_input() -> Tuple[torch.Tensor, int]:
    n_data = random.randint(1, 10)
    x = torch.rand((n_data, 4))
    n_bit = random.randint(1, 8)
    n_boundaries = 2**n_bit
    return x, n_boundaries


@repeat_func(100)
def test_linear_bucketizer() -> None:
    x, n_boundaries = _make_input()

    bucketizer = bucketizer_factory(name="linear")(n_boundaries=n_boundaries)
    ids = bucketizer.encode(x)
    x_cycle = bucketizer.decode(ids)
    ids_cycle = bucketizer.encode(x_cycle)

    assert (torch.abs(x - x_cycle) <= 1 / (2 * n_boundaries)).all()
    assert (ids == ids_cycle).all()


@repeat_func(100)
def test_kmeans_bucketizer() -> None:
    x, n_boundaries = _make_input()
    dataset_name = random.choice(dataset_config_names())
    pkl_name = f"{dataset_name}_kmeans_train_clusters.pkl"
    weight_path = os.path.join(PRECOMPUTED_WEIGHT_DIR, "clustering", pkl_name)

    key = f"{random.choice(GEO_KEYS)}-{n_boundaries}"
    cluster_centers = get_kmeans_cluster_center(weight_path=weight_path, key=key)

    bucketizer = bucketizer_factory(name="kmeans")(
        cluster_centers=cluster_centers, n_boundaries=n_boundaries
    )
    ids = bucketizer.encode(x)
    x_cycle = bucketizer.decode(ids)
    ids_cycle = bucketizer.encode(x_cycle)
    assert (ids == ids_cycle).all()

    # note: no idea on max width of each boundary


if __name__ == "__main__":
    test_linear_bucketizer()
    test_kmeans_bucketizer()
