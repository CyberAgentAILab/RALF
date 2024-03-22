# clustering coordinates for dataset-adaptive tokens (used in LayoutDM)

import argparse
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from image2layout.train.config.dataset import dataset_config_factory
from image2layout.train.data import get_dataset
from omegaconf import OmegaConf
from sklearn.cluster import KMeans

KEYS = ["center_x", "center_y", "width", "height"]
MAIN_KEY = "center_x"

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--result_dir", type=str, default="tmp/weights/clustering")
parser.add_argument("--random_state", type=int, default=0)
parser.add_argument(
    "--max_bbox_num",
    type=int,
    default=int(1e5),
    help="filter number of bboxes to avoid too much time consumption in kmeans",
)

args = parser.parse_args()
dataset_cfg = OmegaConf.structured(dataset_config_factory(args.dataset_name))
n_clusters_list = [2**i for i in range(1, 9)]

dataset_cfg["data_dir"] = args.data_dir

dataset, features = get_dataset(
    dataset_cfg=dataset_cfg,
    transforms=[],
    remove_column_names=["image", "saliency"],
)

_data = defaultdict(list)
for sample in dataset["train"]:
    for key in KEYS:
        _data[key].extend(sample[key])
data = {key: torch.tensor(value) for key, value in _data.items()}

models = {}
weight_path = Path(
    f"{args.result_dir}/{args.dataset_name}_max{dataset_cfg.max_seq_length}_kmeans_train_clusters.pkl"
)
if not weight_path.parent.exists():
    weight_path.parent.mkdir(parents=True, exist_ok=True)

n_samples = data[MAIN_KEY].size(0)
if n_samples > args.max_bbox_num:
    text = f"{n_samples} -> {args.max_bbox_num}"
    logger.warning(
        f"Subsampling bboxes because there are too many for kmeans: ({text})"
    )
    generator = torch.Generator().manual_seed(args.random_state)
    indices = torch.randperm(n_samples, generator=generator)
    data = {key: value[indices[: args.max_bbox_num]] for key, value in data.items()}
    n_samples = args.max_bbox_num

for n_clusters in n_clusters_list:
    start_time = time.time()
    kwargs = {
        "n_clusters": n_clusters,
        "random_state": args.random_state,
        "n_init": "auto",
    }
    for key in KEYS:
        models[f"{key}-{n_clusters}"] = KMeans(**kwargs).fit(
            data[key].numpy()[:, np.newaxis]
        )

    time_elapsed = time.time() - start_time
    logger.info(f"{args.dataset_name}, {n_samples=}, {n_clusters=}, {time_elapsed=}s")

with open(weight_path, "wb") as f:
    pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
