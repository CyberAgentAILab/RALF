import logging
import os
from collections import defaultdict
from typing import Optional

import datasets as ds
import faiss
import fsspec
import numpy as np
import torch
from image2layout.train.helpers.rich_utils import get_progress
from image2layout.train.helpers.util import pad
from numpy import linalg as LA
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

from ..common.base_model import BaseModel, ConditionalInputsForDiscreteLayout
from .image import FeatureExtracterBackbone, coarse_saliency

logger = logging.getLogger(__name__)


class Retriever(BaseModel):
    """
    Copy a layout from db queried by a saliency map similarity.
    Usage:
        bash bin/inference.sh \
            job_dir=image2layout/train/dummy_config/non_learnable \
            result_dir=<RESULT_DIR> +sampling=random cond_type=none
    """

    def __init__(
        self,
        features: ds.Features,
        db_dataset: ds.Dataset,
        max_seq_length: int,
        top_k: int = 1,
        dataset_name: str = "pku",
        retrieval_backbone: str = "saliency",
        saliency_k: Optional[int] = None,
        **kwargs,
    ) -> None:  # type: ignore
        super().__init__()

        self.features = features
        # +1 for no-object
        self.d_label = features["label"].feature.num_classes + 1
        self.max_seq_length = max_seq_length

        self.index_name = "search_feat"
        self.top_k = top_k
        self.dataset_name = dataset_name
        self.retrieval_backbone = retrieval_backbone
        self.db_dataset = db_dataset
        self.output_keys = ["label", "mask", "center_x", "center_y", "width", "height"]

        if retrieval_backbone == "random":
            retrieval_backbone = "saliency"

        if "merge" in retrieval_backbone or "concat" in retrieval_backbone:
            return
        else:
            # Faiss
            self.faiss_index_file_name = (
                f"cache/{dataset_name}_{retrieval_backbone}_wo_head_index.faiss"
            )
            self.backbone = FeatureExtracterBackbone(
                db_dataset=db_dataset, retrieval_backbone=self.retrieval_backbone
            )
            if os.path.exists(self.faiss_index_file_name):
                logger.info(f"Load faiss cache from {self.faiss_index_file_name=}")
                self.db_dataset.load_faiss_index(
                    self.index_name, self.faiss_index_file_name
                )
            else:
                logger.info("Not found faiss cache")

                vectors = self.backbone.extract_dataset_features()
                self.db_dataset.add_faiss_index_from_external_arrays(
                    vectors,
                    index_name=self.index_name,
                    metric_type=faiss.METRIC_INNER_PRODUCT,
                )
                self.db_dataset.save_faiss_index(
                    self.index_name, self.faiss_index_file_name
                )
                logger.info(f"Save cache into {self.faiss_index_file_name}")
                torch.cuda.empty_cache()

    def sample(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = 1,
        sampling_cfg: Optional[DictConfig] = None,
        **kwargs,
    ) -> dict[str, Tensor]:  # type: ignore
        _outputs = {k: [] for k in self.output_keys}  # type: ignore
        B = cond.image.size(0)
        for i in range(B):
            # query = _coarse_saliency(cond.image[i, -1:].cpu())
            if self.retrieval_backbone == "saliency":
                input = cond.image[i, -1:].cpu()
                query = coarse_saliency(input)
            elif self.retrieval_backbone == "random":
                random_idx = np.random.randint(0, len(self.db_dataset))
                saliency = self.db_dataset[random_idx]["saliency"]
                query = coarse_saliency(saliency)
            else:
                input = cond.image[i, :-1]
                query = self.backbone.image_to_feature(input)
            scores, retrieved_examples = self.db_dataset.get_nearest_examples(
                self.index_name, query, k=1
            )
            for key in self.output_keys:
                if key == "mask":
                    n = len(retrieved_examples["label"][0])
                    tensor = torch.tensor(pad([True] * n, self.max_seq_length))
                else:
                    tensor = torch.tensor(
                        pad(retrieved_examples[key][0], self.max_seq_length)
                    )
                _outputs[key].append(tensor)

        empty_violation = {
            "total": 1,
            "viorated": 0,
        }

        return {
            key: torch.stack(_outputs[key], dim=0) for key in _outputs
        }, empty_violation

    def preprocess_retrieval_cache(
        self,
        split: str,
        dataset: str,
        top_k: int,
        run_on_local: bool = True,
        save_scores: bool = False,
    ):
        """
        Calculate retrieval cache among dataset indexes (1:top_k pairs)
        Args:
            top_k: number of retrieved samples
            run_on_local: if True, tqdm is used
        """
        print(f"Start {dataset=}, {split=}, {top_k=}, {run_on_local=}")
        cache_path = f"cache/{self.dataset_name}_{split}_{self.retrieval_backbone}_wo_head_table_between_dataset_indexes_top_k{top_k}.pt"
        fs, path_prefix = fsspec.core.url_to_fs(cache_path)
        # if fs.exists(path_prefix):
        #     logger.info(f"Find the cache in {path_prefix} and loading ...")
        #     table_idx: dict[int, list[int]] = torch.load(cache_path)
        #     return table_idx
        # logger.info(f"Not found cache in {cache_path}")

        logger.info("Calculate retrieval cache among dataset indexes (1:N pairs)")
        cache_table_paired_path = (
            f"cache/{self.dataset_name}_{self.retrieval_backbone}_cache_table_paired.pt"
        )

        if fs.exists(cache_table_paired_path):
            logger.info(f"Load cache from {cache_table_paired_path=}")
            with fs.open(cache_table_paired_path, "rb") as file_obj:
                self.table_paired_id_idx = torch.load(file_obj)
        else:
            pbar = get_progress(
                range(len(self.db_dataset)),
                f"[{split}-{self.retrieval_backbone}] calculate retrieval cache",
                run_on_local,
            )

            self.table_paired_id_idx = {}
            for idx in pbar:
                data_id = self.db_dataset[idx]["id"]
                if "pku" in self.dataset_name:
                    data_id = int(data_id)
                self.table_paired_id_idx[data_id] = idx

            with fs.open(cache_table_paired_path, "wb") as file_obj:
                torch.save(self.table_paired_id_idx, file_obj)

        pbar = tqdm(
            range(len(dataset)),
            desc=f"[{split}-{self.retrieval_backbone}] calculate retrieval cache",
            total=len(dataset),
        )

        table_data_id_to_dataset_idx = defaultdict(list)
        if save_scores:
            table_data_id_to_scores = defaultdict(list)

        for idx in pbar:
            data: dict[str, Tensor] = dataset[idx]
            query: np.ndarray = self.backbone.get_query(data)
            data_id = data["id"]
            if "pku" in self.dataset_name:
                data_id = int(data_id)

            scores, retrieved_examples = self.db_dataset.get_nearest_examples(
                self.index_name, query, k=top_k + 1
            )
            retrieved_ids: list[int] = retrieved_examples["id"]
            if "pku" in self.dataset_name:
                retrieved_ids: list[int] = list(map(int, retrieved_ids))
            retrieved_idxs: list[int] = [
                self.table_paired_id_idx[_id] for _id in retrieved_ids
            ]

            # remove itself if split is train
            if split == "train":
                retrieved_idxs = retrieved_idxs[1:]
                scores = scores[1:]

            table_data_id_to_dataset_idx[data_id] = retrieved_idxs
            if save_scores:
                table_data_id_to_scores[data_id] = scores

        with fs.open(cache_path, "wb") as file_obj:
            logger.info(f"Save cache into {cache_path}")
            torch.save(table_data_id_to_dataset_idx, file_obj)

        if save_scores:
            score_cache_path = cache_path.replace("indexes", "scores")
            with fs.open(score_cache_path, "wb") as file_obj:
                logger.info(f"Save cache into {score_cache_path=}")
                torch.save(table_data_id_to_scores, file_obj)

        return table_data_id_to_dataset_idx

    def preprocess_to_merge_retrieval_cache(
        self, dataset_name, split, dataset, top_k, run_on_local, where_norm
    ):
        """
        Calculate retrieval cache among dataset indexes (1:top_k pairs)
        Args:
            top_k: number of retrieved samples
            run_on_local: if True, tqdm is used
        """
        print(f"Start {dataset=}, {split=}, {top_k=}, {run_on_local=}")

        retrieval_backbones = self.retrieval_backbone.split("_")[1:]
        save_db_path = f"{dataset_name}_{self.retrieval_backbone}.faiss"
        fs, path_prefix = fsspec.core.url_to_fs(save_db_path)

        # Merge cache
        if fs.exists(save_db_path):
            logger.info(f"Find the cache in {save_db_path} and loading ...")
            self.db_dataset.load_faiss_index(self.index_name, save_db_path)
        else:
            logger.info(f"Not found cache in {save_db_path}")

            # 1. Load cache
            vectors = []
            for _backbone in retrieval_backbones:
                cache_path = f"{dataset_name}_{_backbone}_wo_head_index.faiss"
                _db = faiss.read_index(cache_path)
                _vec = _db.reconstruct_n(0, _db.ntotal)
                if where_norm == "before_concat":
                    _vec = _vec / LA.norm(_vec, ord=2)
                vectors.append(_vec)

            # 2. Merge cache
            vectors = np.concatenate(vectors, axis=1)
            if where_norm == "after_concat":
                # TODO: Ablation. Pattern2: normalize after concat
                vectors = vectors / LA.norm(vectors, ord=2)

            # 3. Register to db_dataset
            self.db_dataset.add_faiss_index_from_external_arrays(
                vectors,
                index_name=self.index_name,
                metric_type=faiss.METRIC_INNER_PRODUCT,
            )
            self.db_dataset.save_faiss_index(self.index_name, save_db_path)
            logger.info(f"Save cache into {save_db_path}")

        cache_dataid_dbidx_path = (
            f"cache/{self.dataset_name}_train_saliency_cache_table_paired.pt"
        )
        with fs.open(cache_dataid_dbidx_path, "rb") as file_obj:
            table_dataid_dbidx = torch.load(file_obj)
            # Key: data_id, Value: db_idx

        cache_db_path = f"cache/{dataset_name}_{split}_{self.retrieval_backbone}_{where_norm}__topk{top_k}.pt"
        # if fs.exists(cache_db_path):
        #     logger.info(f"Find the cache in {cache_db_path}")
        #     return

        pbar = tqdm(
            range(len(dataset)),
            desc=f"[{split}-{self.retrieval_backbone}] calculate retrieval cache",
            total=len(dataset),
        )

        backbones = {}
        for _backbone in retrieval_backbones:
            backbones[_backbone] = FeatureExtracterBackbone(
                db_dataset=self.db_dataset, retrieval_backbone=_backbone
            )

        table_data_id_to_dataset_idx = defaultdict(list)
        for idx in pbar:
            data: dict[str, Tensor] = dataset[idx]
            queries = []

            for _backbone in retrieval_backbones:
                query: np.ndarray = backbones[_backbone].get_query(data)
                if where_norm == "before_concat":
                    # TODO: Ablation. Pattern1: normalize before concat
                    query = query / LA.norm(query, ord=2)
                queries.append(query)

            query = np.concatenate(queries, axis=0)
            if where_norm == "after_concat":
                # TODO: Ablation. Pattern2: normalize after concat
                query = query / LA.norm(query, ord=2)
            data_id = data["id"]
            if "pku" in self.dataset_name:
                data_id = int(data_id)

            scores, retrieved_examples = self.db_dataset.get_nearest_examples(
                self.index_name, query, k=top_k + 1
            )
            retrieved_ids: list[int] = retrieved_examples["id"]
            if "pku" in self.dataset_name:
                retrieved_ids: list[int] = list(map(int, retrieved_ids))
            retrieved_idxs: list[int] = [
                table_dataid_dbidx[_id] for _id in retrieved_ids
            ]

            # remove itself if split is train
            if split == "train":
                retrieved_idxs = retrieved_idxs[1:]

            table_data_id_to_dataset_idx[data_id] = retrieved_idxs

            # if idx > 50:
            #     break

        with fs.open(cache_db_path, "wb") as file_obj:
            logger.info(f"Save cache into {cache_db_path}")
            torch.save(table_data_id_to_dataset_idx, file_obj)
