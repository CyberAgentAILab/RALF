import argparse
import os

from image2layout.train.config import get_mock_train_cfg
from image2layout.train.data import get_dataset
from image2layout.train.models.retrieval.retriever import Retriever

DATASETS = ["pku", "cgl"]
RETRIEVAL_BACKBONES = ["saliency", "clip", "vgg"]


def main():
    """
    Pre-compute and cache indexes (and optionally similarity scores) for nearest neighbour search.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pku", choices=DATASETS)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument(
        "--retrieval_backbone",
        type=str,
        default="dreamsim",
        choices=RETRIEVAL_BACKBONES,
    )
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="some reranking methods needs similarity scores between query and retrieved data",
    )
    args = parser.parse_args()

    preprocess_retriever(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        retrieval_backbone=args.retrieval_backbone,
        top_k=args.top_k,
        save_scores=args.save_scores,
    )


def preprocess_retriever(
    dataset_path: str = "/datasets/PosterLayout",
    dataset_name: str = "pku",
    max_seq_length: int = 10,
    retrieval_backbone: str = "saliency",
    top_k: int = 32,
    save_scores: bool = False,
) -> None:
    if dataset_name == "pku":
        _data_dir = f"{dataset_name}{max_seq_length}"
    else:
        _data_dir = dataset_name

    train_cfg = get_mock_train_cfg(
        max_seq_length, os.path.join(dataset_path, _data_dir)
    )

    datasets, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )

    retriever = Retriever(
        features=features,
        db_dataset=datasets["train"],
        max_seq_length=max_seq_length,
        dataset_name=dataset_name,
        retrieval_backbone=retrieval_backbone,
    )

    for split in datasets.keys():
        retriever.preprocess_retrieval_cache(
            split=split,
            dataset=datasets[split],
            top_k=top_k,
            run_on_local=True,
            save_scores=save_scores,
        )


def preprocess_merged_retriever(
    dataset_path: str = "/datasets/PosterLayout",
    dataset_name: str = "pku",
    max_seq_length: int = 10,
    retrieval_backbone: str = "clip",
    where_norm: str = "after_concat",
    top_k: int = 32,
):
    if dataset_name == "pku":
        _data_dir = f"{dataset_name}{max_seq_length}"
    else:
        _data_dir = dataset_name
    train_cfg = get_mock_train_cfg(
        max_seq_length, os.path.join(dataset_path, _data_dir)
    )

    datasets, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )

    retriever = Retriever(
        features=features,
        db_dataset=datasets["train"],
        max_seq_length=max_seq_length,
        dataset_name=dataset_name,
        retrieval_backbone=retrieval_backbone,
    )

    for split in datasets.keys():
        retriever.preprocess_to_merge_retrieval_cache(
            dataset_name=dataset_name,
            split=split,
            dataset=datasets[split],
            top_k=top_k,
            run_on_local=True,
            where_norm=where_norm,
        )


if __name__ == "__main__":
    main()

# OMP_NUM_THREADS=2 poetry run python3 image2layout/preprocess/build_retrieval_indexes.py --dataset pku --dataset_path /datasets/PosterLayout
