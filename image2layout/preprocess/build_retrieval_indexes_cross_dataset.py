import argparse
import os

from image2layout.train.config import get_mock_train_cfg
from image2layout.train.data import get_dataset
from image2layout.train.models.retrieval.cross_retriever import CrossRetriever

DATASETS = ["pku", "cgl"]
RETRIEVAL_BACKBONES = ["saliency", "clip", "vgg"]


def main():
    """
    Pre-compute and cache indexes (and optionally similarity scores) for nearest neighbour search.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument(
        "--retrieval_backbone",
        type=str,
        default="dreamsim",
        choices=RETRIEVAL_BACKBONES,
    )
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="some reranking methods needs similarity scores between query and retrieved data",
    )
    args = parser.parse_args()

    preprocess_cross_retriever(
        dataset_path=args.dataset_path,
        retrieval_backbone=args.retrieval_backbone,
        top_k=args.top_k,
        save_scores=args.save_scores,
    )


def preprocess_cross_retriever(
    dataset_path: str = "/datasets/PosterLayout",
    max_seq_length: int = 10,
    retrieval_backbone: str = "saliency",
    top_k: int = 32,
    save_scores: bool = False,
) -> None:

    train_cfg_pku = get_mock_train_cfg(
        max_seq_length, os.path.join(dataset_path, "pku10")
    )
    train_cfg_cgl = get_mock_train_cfg(
        max_seq_length, os.path.join(dataset_path, "cgl")
    )

    datasets_pku, features_pku = get_dataset(
        dataset_cfg=train_cfg_pku.dataset,
        transforms=list(train_cfg_pku.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )

    datasets_cgl, features_cgl = get_dataset(
        dataset_cfg=train_cfg_cgl.dataset,
        transforms=list(train_cfg_cgl.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )

    cross_retriever = CrossRetriever(
        features_pku=features_pku,
        features_cgl=features_cgl,
        db_dataset_pku=datasets_pku["train"],
        db_dataset_cgl=datasets_cgl["train"],
        max_seq_length=max_seq_length,
        retrieval_backbone=retrieval_backbone,
    )

    split = "with_no_annotation"

    cross_retriever.preprocess_retrieval_cache(
        split=split,
        source="pku",
        reference="cgl",
        dataset_source=datasets_pku[split],
        dataset_reference=datasets_cgl[split],
        top_k=top_k,
        run_on_local=True,
        save_scores=save_scores,
    )

    cross_retriever.preprocess_retrieval_cache(
        split=split,
        source="cgl",
        reference="pku",
        dataset_source=datasets_cgl[split],
        dataset_reference=datasets_pku[split],
        top_k=top_k,
        run_on_local=True,
        save_scores=save_scores,
    )


if __name__ == "__main__":
    main()

# OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=4 poetry run python3 image2layout/preprocess/build_retrieval_indexes_cross_dataset.py --dataset_path /datasets/PosterLayout
[]
