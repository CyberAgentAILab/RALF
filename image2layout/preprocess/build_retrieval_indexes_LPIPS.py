import argparse
import os

import lpips
import torch
from image2layout.train.config import get_mock_train_cfg
from image2layout.train.data import get_dataset
from image2layout.train.models.retrieval.retriever import Retriever
from tqdm import tqdm

DATASETS = ["pku", "cgl"]
RETRIEVAL_BACKBONES = ["saliency", "clip", "vgg"]


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


# Learned perceptual metric
class CacheLPIPS(lpips.LPIPS):
    def get_embedding(self, in0, normalize=False):
        if normalize:
            in0 = 2 * in0 - 1

        if self.version == "0.1":
            in0_input = self.scaling_layer(in0)
        else:
            in0_input = in0

        outs0 = self.net.forward(in0_input)

        feats0 = {}
        for kk in range(self.L):
            feats0[kk] = lpips.normalize_tensor(outs0[kk])
        return feats0

    def calculate_diffs(self, feats0, feats1):
        diffs = {}
        for kk in range(self.L):
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
            for kk in range(self.L)
        ]

        val = 0
        for l in range(self.L):
            val += res[l]

        return val


def main():
    """
    Pre-compute and cache indexes (and optionally similarity scores) for nearest neighbour search.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pku", choices=DATASETS)
    parser.add_argument("--dataset_path", type=str)
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
        top_k=args.top_k,
        save_scores=args.save_scores,
    )


def preprocess_retriever(
    dataset_path: str = "/datasets/PosterLayout",
    dataset_name: str = "pku",
    max_seq_length: int = 10,
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

    LPIPS = CacheLPIPS(net="alex").cuda()

    # Calculate database cache
    DATABASE = {}
    for j in tqdm(range(len(datasets["train"])), desc=f"Caching database"):
        data_entitiy = datasets["train"][j]
        img_emtity = data_entitiy["image"].unsqueeze(0).cuda()  # [1, 3, H, W], [0, 1]
        id_emtity = data_entitiy["id"]

        with torch.cuda.amp.autocast(enabled=True):
            feat1 = LPIPS.get_embedding(img_emtity, normalize=True)
        feat1 = {k: v.cpu() for k, v in feat1.items()}
        DATABASE[j] = feat1

        # if j > 10:
        #     break

    for split in datasets.keys():
        N = len(datasets[split])
        print(f"Split: {split}, Num: {N}")
        RESULT = {}
        for i in tqdm(range(N)):
            data_query = datasets[split][i]
            img_query = data_query["image"].unsqueeze(0).cuda()  # [1, 3, H, W], [0, 1]
            id_query = data_query["id"]
            if dataset_name == "pku":
                id_query = int(id_query)

            with torch.cuda.amp.autocast(enabled=True):
                feat0 = LPIPS.get_embedding(img_query, normalize=True)

                SCORE = {}
                for j, feat1 in DATABASE.items():

                    if split == "train" and j == i:
                        continue

                    feat1 = {k: v.cuda() for k, v in feat1.items()}
                    score = LPIPS.calculate_diffs(feat0, feat1)
                    SCORE[j] = score.item()

            sorted_SCORE = sorted(SCORE.items(), key=lambda item: item[1])
            top_k_ids = [int(a) for a, _ in sorted_SCORE[:top_k]]
            RESULT[id_query] = top_k_ids

        cache_path = f"cache/{dataset_name}_{split}_lpips_wo_head_table_between_dataset_indexes_top_k{top_k}.pt"
        torch.save(RESULT, cache_path)
        print(f"Save: {cache_path}")

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 poetry run python3 image2layout/preprocess/build_retrieval_indexes_LPIPS.py --dataset_name pku --dataset_path /datasets/PosterLayout

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 poetry run python3 image2layout/preprocess/build_retrieval_indexes_LPIPS.py --dataset_name cgl --dataset_path /datasets/PosterLayout
