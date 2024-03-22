import logging
import os
from collections import defaultdict
from typing import Optional

import datasets as ds
import faiss
import fsspec
import numpy as np
import torch
from image2layout.train.data import pad
from image2layout.train.helpers.rich_utils import get_progress
from numpy import linalg as LA
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

from ..common.base_model import BaseModel, ConditionalInputsForDiscreteLayout
from .image import FeatureExtracterBackbone, coarse_saliency

logger = logging.getLogger(__name__)


class CrossRetriever(BaseModel):
    """
    Copy a layout from db queried by a saliency map similarity.
    Usage:
        bash bin/inference.sh \
            job_dir=image2layout/train/dummy_config/non_learnable \
            result_dir=<RESULT_DIR> +sampling=random cond_type=none
    """

    def __init__(
        self,
        features_pku: ds.Features,
        features_cgl: ds.Features,
        db_dataset_pku: ds.Dataset,
        db_dataset_cgl: ds.Dataset,
        max_seq_length: int,
        top_k: int = 1,
        retrieval_backbone: str = "saliency",
        saliency_k: Optional[int] = None,
        **kwargs,
    ) -> None:  # type: ignore
        super().__init__()

        self.features_pku = features_pku
        self.features_cgl = features_cgl
        self.max_seq_length = max_seq_length

        self.index_name = "search_feat"
        self.top_k = top_k
        self.retrieval_backbone = retrieval_backbone
        self.db_dataset_pku = db_dataset_pku
        self.db_dataset_cgl = db_dataset_cgl
        self.output_keys = ["label", "mask", "center_x", "center_y", "width", "height"]

        if retrieval_backbone == "random":
            retrieval_backbone = "saliency"

        if "merge" in retrieval_backbone or "concat" in retrieval_backbone:
            return
        else:
            self.backbone_pku = FeatureExtracterBackbone(
                db_dataset=db_dataset_pku, retrieval_backbone=self.retrieval_backbone
            )
            self.backbone_cgl = FeatureExtracterBackbone(
                db_dataset=db_dataset_cgl, retrieval_backbone=self.retrieval_backbone
            )
            self.load_faiss_index("pku", retrieval_backbone, self.db_dataset_pku)
            self.load_faiss_index("cgl", retrieval_backbone, self.db_dataset_cgl)

    def load_faiss_index(self, dataset_name, retrieval_backbone, db_dataset):
        faiss_index_file_name = (
            f"cache/{dataset_name}_{retrieval_backbone}_wo_head_index.faiss"
        )
        db_dataset.load_faiss_index(self.index_name, faiss_index_file_name)

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

            # quick visualization to see what are
            # import torchvision.utils as vutils
            # from image2layout.train.helpers.visualizer import render
            # query_image, query_saliency = cond.image[i, :-1], cond.image[i, -1:]
            # vis_example = {k: v[0] for (k, v) in outputs.items()}
            # vis_example["image"] = query_image
            # output = render(vis_example, self.features["label"].feature, bg_key="image")
            # vutils.save_image(output, f"prediction_{i}.png")
            # vutils.save_image(query_image, f"image_{i}.png")
            # vutils.save_image(query_saliency, f"saliency_{i}.png")

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
        source: str,
        reference: str,
        dataset_source,
        dataset_reference,
        top_k: int,
        run_on_local: bool = True,
        save_scores: bool = False,
    ):
        """
        Calculate retrieval cache between cgl -> pku or pku -> cgl (1:top_k pairs)
        Args:
            top_k: number of retrieved samples
            run_on_local: if True, tqdm is used
        """

        self.dataset_name = f"source_{source}_reference_{reference}"
        cache_path = f"cache/{self.dataset_name}_{split}_{self.retrieval_backbone}_cross_dataset_indexes_top_k{top_k}.pt"
        logger.info(f"{cache_path=}")
        fs, path_prefix = fsspec.core.url_to_fs(cache_path)

        logger.info("Calculate retrieval cache among dataset indexes (1:N pairs)")
        self.table_paired_id_idx = {}
        for dataset in ["pku", "cgl"]:

            cache_table_paired_path = (
                f"cache/{dataset}_{self.retrieval_backbone}_cache_table_paired.pt"
            )
            logger.info(f"Load cache from {cache_table_paired_path=}")
            with fs.open(cache_table_paired_path, "rb") as file_obj:
                self.table_paired_id_idx[dataset] = torch.load(file_obj)

        if source == "pku":
            backbone = self.backbone_pku
            db_dataset_reference = self.db_dataset_cgl
        elif source == "cgl":
            backbone = self.backbone_cgl
            db_dataset_reference = self.db_dataset_pku

        pbar = tqdm(
            range(len(dataset_source)),
            desc=f"[{source=}, {reference=}] calculate retrieval cache",
            total=len(dataset_source),
        )

        table_data_id_to_dataset_idx = defaultdict(list)
        if save_scores:
            table_data_id_to_scores = defaultdict(list)

        for idx in pbar:
            data_source: dict[str, Tensor] = dataset_source[idx]
            query: np.ndarray = backbone.get_query(data_source)
            data_source_id = data_source["id"]
            if "pku" in dataset_source:
                data_source_id = int(data_source_id)

            scores, retrieved_examples = db_dataset_reference.get_nearest_examples(
                self.index_name, query, k=top_k + 1
            )
            retrieved_ids: list[int] = retrieved_examples["id"]
            if "pku" in reference:
                retrieved_ids: list[int] = list(map(int, retrieved_ids))
            retrieved_idxs: list[int] = [
                self.table_paired_id_idx[reference][_id] for _id in retrieved_ids
            ]

            table_data_id_to_dataset_idx[data_source_id] = retrieved_idxs
            if save_scores:
                table_data_id_to_scores[data_source_id] = scores

        with fs.open(cache_path, "wb") as file_obj:
            logger.info(f"Save cache into {cache_path}")
            torch.save(table_data_id_to_dataset_idx, file_obj)

        # if save_scores:
        #     score_cache_path = cache_path.replace("indexes", "scores")
        #     with fs.open(score_cache_path, "wb") as file_obj:
        #         logger.info(f"Save cache into {score_cache_path=}")
        #         torch.save(table_data_id_to_scores, file_obj)

        # return table_data_id_to_dataset_idx
