import logging
import os
import random
from typing import Optional, Union

import datasets as ds
import fsspec
import numpy as np
import torch
from image2layout.train.global_variables import PRECOMPUTED_WEIGHT_DIR
from image2layout.train.helpers.util import pad
from torch import Tensor

logger = logging.getLogger(__name__)


def load_cache_table(cache_path: str, top_k: int) -> dict[int, list[int]]:
    fs, path_prefix = fsspec.core.url_to_fs(cache_path)
    if fs.exists(path_prefix):
        logger.info(f"Find {cache_path=} and loading ...")
    else:
        cache_path = os.path.join(
            PRECOMPUTED_WEIGHT_DIR, "retrieval_indexes", path_prefix.split("/")[-1]
        )
        fs, path_prefix = fsspec.core.url_to_fs(cache_path)
        if not fs.exists(path_prefix):
            raise ValueError(f"Cache not found in {path_prefix}")
        logger.info(f"Find {cache_path=} and loading ...")

    with fs.open(path_prefix, "rb") as f:
        table_idx: dict[int, list[int]] = torch.load(f)
    table_idx = {k: v[:top_k] for k, v in table_idx.items()}
    return table_idx


class RetrievalDatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset: ds.Dataset,
        db_dataset: ds.Dataset,
        split: str,
        top_k: int,
        max_seq_length: int,
        retrieval_backbone: str,
        random_retrieval: bool,
        saliency_k: Union[int, str],
        num_cache_indexes_per_sample: int = 32,  # too many indexes results in slow preprocessing
        inference_num_saliency: Optional[int] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.db_dataset = db_dataset
        self.split = split
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        self.retrieval_backbone = retrieval_backbone

        cache_path = f"cache/{self.dataset_name}_{split}_{retrieval_backbone}_wo_head_table_between_dataset_indexes_top_k{num_cache_indexes_per_sample}.pt"
        self.table_idx = load_cache_table(cache_path, top_k)
        logger.info(f"Load single db from {cache_path}!")

        self.keys = [
            "image",
            "saliency",
            "center_x",
            "center_y",
            "width",
            "height",
            "label",
            "mask",
        ]

    def load_faiss_index(self, *args, **kwargs):
        return self.dataset.load_faiss_index(*args, **kwargs)

    def get_nearest_examples(self, *args, **kwargs):
        return self.dataset.get_nearest_examples(*args, **kwargs)

    def add_faiss_index_from_external_arrays(self, *args, **kwargs):
        return self.dataset.add_faiss_index_from_external_arrays(*args, **kwargs)

    def save_faiss_index(self, *args, **kwargs):
        return self.dataset.save_faiss_index(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:

        data = self.dataset[index]

        data_ids = data["id"]
        if not isinstance(data_ids, list):
            data_ids = [data["id"]]

        data["retrieved"] = []
        for data_id in data_ids:

            retrieved = {}

            if "pku" in self.dataset_name:
                data_id = int(data_id)

            if isinstance(self.table_idx, list):
                retrieved_indexes: list = []
                for _table in self.table_idx:
                    retrieved_indexes += _table[data_id]

            else:
                retrieved_indexes: list = self.table_idx[data_id]

            assert (
                len(retrieved_indexes) == self.top_k
            ), f"{len(retrieved_indexes)=} != {self.top_k=}"
            retrieved_data = [self.db_dataset[_idx] for _idx in retrieved_indexes]
            assert len(retrieved_data) == self.top_k
            retrieved["index"] = retrieved_indexes
            for key in self.keys:
                if key == "mask":
                    tensor = torch.tensor(
                        [
                            pad(
                                [True] * len(retrieved_data[i]["label"]),
                                self.max_seq_length,
                            )
                            for i in range(self.top_k)
                        ]
                    )
                elif isinstance(retrieved_data[0][key], list):
                    tensor = torch.tensor(
                        [
                            pad(retrieved_data[i][key], self.max_seq_length)
                            for i in range(self.top_k)
                        ]
                    )  # [K, max_elem]
                elif isinstance(retrieved_data[0][key], Tensor):
                    # [K, N, H, W][]
                    tensor = torch.stack(
                        [retrieved_data[i][key] for i in range(self.top_k)]
                    )

                assert tensor.size(0) == self.top_k
                retrieved[key] = tensor

            data["retrieved"].append(retrieved)

        return data
