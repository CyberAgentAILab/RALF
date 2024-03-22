import argparse
import logging
import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from image2layout.train.config import get_mock_train_cfg
from image2layout.train.data import collate_fn, get_dataset
from image2layout.train.fid.model import FIDNetV3, load_fidnet_v3
from image2layout.train.helpers.retrieval_dataset_wrapper import RetrievalDatasetWrapper
from image2layout.train.helpers.rich_utils import CONSOLE
from image2layout.train.models.retrieval.reranker import (
    maximal_marginal_relevance,
    reranker_random,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


SPLITS = ["train", "val", "test", "with_no_annotation"]


@torch.no_grad()
def main(args):
    if args.dataset == "pku":
        data_dir = os.path.join(
            args.dataset_path, f"{args.dataset}{args.max_seq_length}"
        )
    else:
        data_dir = os.path.join(args.dataset_path, args.dataset)
    train_cfg = get_mock_train_cfg(args.max_seq_length, data_dir)

    # Build dataset
    dataset, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
    )

    # Build loaders
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    collate_fn_partial = partial(collate_fn, max_seq_length=args.max_seq_length)

    for split in SPLITS:
        _dataset = RetrievalDatasetWrapper(
            dataset_name=args.dataset,
            dataset=dataset[split],
            db_dataset=dataset["train"],
            split=split,
            top_k=args.rerank_pool_size,
            num_cache_indexes_per_sample=args.rerank_pool_size,
            max_seq_length=args.max_seq_length,
            retrieval_backbone=args.retrieval_backbone,
            random_retrieval=False,
            saliency_k=0,
            inference_num_saliency=0,
        )
        loaders[split] = torch.utils.data.DataLoader(
            _dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,
            pin_memory=True,
            collate_fn=collate_fn_partial,
            persistent_workers=False,
            drop_last=False,
        )

    # Build Layout Feature Extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_label = features["label"].feature
    fid_model_layout = FIDNetV3(
        num_label=feature_label.num_classes,
        max_bbox=train_cfg.dataset.max_seq_length,
    )
    logger.info(f"Loading FIDNetV3 model from {args.fid_weight_dir} ...")
    fid_model_layout = load_fidnet_v3(fid_model_layout, args.fid_weight_dir).to(device)

    name = f"{args.dataset}_{split}_{args.retrieval_backbone}_wo_head_table_between_dataset_indexes_"
    name += f"top_k{args.top_k}_by_{args.rerank_type}_from_{args.rerank_pool_size}.pt"

    CosSim = nn.CosineSimilarity(dim=1, eps=1e-08)

    for split in ["train", "val", "test"]:
        indexes_path = f"cache/{args.dataset}_{split}_{args.retrieval_backbone}_wo_head_table_between_dataset_indexes_top_k{args.rerank_pool_size}.pt"
        table_indexes = torch.load(indexes_path)
        table_scores = torch.load(indexes_path.replace("indexes", "scores"))

        if args.rerank_type == "mmr":
            params_str = f"rerank_{args.rerank_type}_lam_{args.rerank_mmr_lam}"
        elif args.rerank_type == "random":
            params_str = f"rerank_{args.rerank_type}"
        else:
            raise NotImplementedError
        reranked_indexes_path = f"cache/{args.dataset}_{split}_{args.retrieval_backbone}_{params_str}_wo_head_table_between_dataset_indexes_top_k{args.top_k}.pt"
        table_indexes_reranked = defaultdict(list)

        loader = loaders[split]
        for idx, batch in enumerate(
            tqdm(loader, desc=f"{split=}, rerank_type={args.rerank_type}")
        ):
            """Visuzalition of input & retrieved layouts"""
            _dataid = batch["id"][0]
            if "pku" in args.dataset:
                _dataid = int(_dataid)
            retrieved = batch["retrieved"][0]
            retrieved_tensor = {
                k: v.to(device)[0]
                for k, v in retrieved.items()
                if isinstance(v, torch.Tensor)
            }

            retrieved_feat = fid_model_layout.extract_features(
                retrieved_tensor
            )  # [K, 256]
            K = len(retrieved_feat)

            if args.rerank_type == "random":
                local_indexes = reranker_random(K, args.top_k)
            else:
                score_di_q = table_scores[_dataid][:K]

                di = rearrange(retrieved_feat, "s d -> s d 1")
                dj = rearrange(retrieved_feat, "s d -> 1 d s")
                score_di_dj = CosSim(di, dj).cpu()

                local_indexes = maximal_marginal_relevance(
                    score_di_q=score_di_q,
                    score_di_dj=np.array(score_di_dj),
                    top_k=args.top_k,
                    score_type="similarity",
                    lam=args.rerank_mmr_lam,
                )  # note: valid only inside this batch!
                print(local_indexes.mean())

            table_indexes_reranked[_dataid] = np.array(table_indexes[_dataid])[
                local_indexes
            ].tolist()

        logger.info(f"Saving reranked results to {reranked_indexes_path} ...")
        torch.save(
            table_indexes_reranked,
            reranked_indexes_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--max_seq_length", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="pku")
    parser.add_argument("--dataset_path", type=str, default="/datasets/PosterLayout")
    parser.add_argument("--top_k", type=int, default=32)

    parser.add_argument(
        "--retrieval_backbone",
        type=str,
        default="dreamsim",
    )
    parser.add_argument("--rerank_pool_size", type=int, default=128)
    parser.add_argument(
        "--rerank_type",
        type=str,
        default="mmr",
        choices=["mmr", "random"],
    )
    parser.add_argument("--rerank_mmr_lam", type=float, default=1.0)

    # FID
    parser.add_argument("--fid_weight_dir", type=str, default="tmp/fidnet/pku10")

    args = parser.parse_args()
    CONSOLE.print(args)

    main(args)

# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 poetry run python image2layout/preprocess/rerank_indexes.py --rerank_type mmr --rerank_mmr_lam 0.99 --dataset_path /home/jupyter/datasets/image_conditioned_layout_generation/preprocessed_parquets --rerank_pool_size 64 --dataset cgl --fid_weight_dir tmp/fidnet/cgl --dataset cgl
