import logging
from typing import Optional, Union

import datasets as ds
import torch
from image2layout.train.helpers.util import pad
from torch import Tensor

logger = logging.getLogger(__name__)


class RandomRetrievalDatasetWrapper(torch.utils.data.Dataset):
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
        self.saliency_k = saliency_k

        logger.info(f"Use retrieval dataset with {saliency_k=}")

        random_retrieval = True
        self.random_retrieval = random_retrieval

        logger.info(f"Warning!!! Random retrieval for {dataset_name}--{split}!!!")

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

            retrieved_indexes: list = torch.randint(
                low=0, high=self.__len__(), size=[self.top_k]
            ).tolist()

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
