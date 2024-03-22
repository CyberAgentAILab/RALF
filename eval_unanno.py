import argparse
import copy
import logging
import os
from collections import defaultdict
from functools import partial

import fsspec
import torch
import yaml
from eval import KEYS, compute_average, load_pkl
from image2layout.train.data import collate_fn, get_dataset
from image2layout.train.helpers.metric import (
    compute_alignment,
    compute_overlap,
    compute_overlay,
    compute_rshm,
    compute_saliency_aware_metrics,
    compute_underlay_effectiveness,
    compute_validity,
)
from image2layout.train.helpers.rich_utils import get_progress
from image2layout.train.helpers.util import set_seed
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--load-gt-split",
        type=str,
        choices=["val", "test"],
        default=None,
        help="instead of loading generated samples, load ground truth samples from the specified split",
    )
    parser.add_argument(
        "--save-score-dir",
        type=str,
        default="tmp/scores",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    set_seed(0)

    if args.debug:
        logger.info("Debug mode!")

    # Create result directory
    fs, path_prefix = fsspec.core.url_to_fs(args.save_score_dir)
    if not fs.exists(path_prefix):
        fs.makedirs(path_prefix)

    use_generated_samples = args.load_gt_split is None

    if use_generated_samples:
        # Load all pickle files
        fs, _ = fsspec.core.url_to_fs(args.input_dir)
        scores_all_path = os.path.join(args.input_dir, "scores_all.yaml")
        # if fs.exists(scores_all_path):
        #     logger.info(f"Find {scores_all_path}. Finish!")
        #     return None
        pickle_paths = fs.glob(os.path.join(args.input_dir, "*.pkl"))
        logger.info(f"Found pickle files: {pickle_paths=}")

    else:
        pickle_paths = [None]
        ckpt_name = "ground-truth dataset"
        seed = "None"
        split = args.load_gt_split

        train_cfg = OmegaConf.create(
            {
                "dataset": {
                    "max_seq_length": 10,
                    "data_dir": args.dataset_path,
                    "data_type": "parquet",
                    "path": None,
                },
                "data": {"transforms": ["image", "shuffle"], "tokenization": False},
                "run_on_local": True,
            }
        )

        test_cfg = OmegaConf.create(
            {
                "dataset": {
                    "max_seq_length": 10,
                    "data_dir": args.dataset_path,
                    "data_type": "parquet",
                },
                "batch_size": 1,
                "dataset_path": args.dataset_path,
            }
        )
        logger.info(f"Use ground-truth {split=} dataset")

    # Build dataset
    if use_generated_samples:
        train_cfg, test_cfg = load_pkl(pickle_paths[0])[2:4]

    training_data_dir = train_cfg.dataset.data_dir
    dataset_cfg = copy.deepcopy(train_cfg.dataset)
    dataset_cfg.data_dir = args.dataset_path

    dataset, features = get_dataset(
        dataset_cfg=dataset_cfg,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )

    # Check whether a cross-evaluation setting
    training_dataset_name = train_cfg.dataset.data_dir.split("/")[-1][:3]
    eval_dataset_name = args.dataset_path.split("/")[-1][:3]
    use_cross_dataset = False
    if training_dataset_name != eval_dataset_name:
        use_cross_dataset = True
        dataset_cfg.data_dir = training_data_dir
        _, features = get_dataset(
            dataset_cfg=dataset_cfg,
            transforms=list(train_cfg.data.transforms),
            remove_column_names=["image_width", "image_height"],
        )

    # Build dataloader
    max_seq_length = train_cfg.dataset.max_seq_length
    if max_seq_length < 0:
        max_seq_length = None
    collate_fn_partial = partial(
        collate_fn,
        max_seq_length=max_seq_length,
    )
    loaders = {}
    batch_size = test_cfg.batch_size
    for _split in ["with_no_annotation"]:
        loaders[_split] = torch.utils.data.DataLoader(
            dataset[_split],
            num_workers=2,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=collate_fn_partial,
            persistent_workers=False,
            drop_last=False,
            shuffle=False,
        )

    # Build metrics
    feature_label = features["label"].feature
    batch_eval_funcs = [
        compute_alignment,
        compute_overlap,
        partial(compute_saliency_aware_metrics, feature_label=feature_label),
        partial(compute_overlay, feature_label=feature_label),
        partial(compute_underlay_effectiveness, feature_label=feature_label),
        compute_rshm,
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scores_all = defaultdict(list)

    for pickle_path in pickle_paths:
        # Load picjke
        if use_generated_samples:
            (
                fs,
                generated_samples,
                train_cfg,
                test_cfg,
                _,
                _,
                ckpt_name,
            ) = load_pkl(pickle_path)
            split = "with_no_annotation"
            seed = (
                pickle_path.split("/")[-1]
                .split(".pkl")[0]
                .split("with_no_annotation_")[-1]
            )

        else:
            # Load ground truth samples for gt-gt evaluation
            generated_samples = [
                {k: v for k, v in dataset[split][i].items() if k in KEYS}
                for i in range(len(dataset[split]))
            ]

        generated_samples, validity = compute_validity(generated_samples)

        # Attach image and saliency to generated samples.
        assert len(dataset[split]) == len(
            generated_samples
        ), f"{len(dataset[split])} != {len(generated_samples)}"

        # compute scores for each run
        logger.info("Evaluation start!!")

        batch_metrics = defaultdict(list)

        # Compute metrics and extract features.
        pbar = get_progress(
            range(0, len(generated_samples), batch_size),
            "Eval generated samples",
        )
        for i in pbar:
            i_end = min(i + batch_size, len(generated_samples))
            _batch = generated_samples[i:i_end]

            # append image and saliency in batch-wise manner to avoid OOM
            for j in range(i, i_end):
                assert _batch[j - i]["id"] == dataset[split][j]["id"]
                for key in ["image", "saliency"]:
                    _batch[j - i][key] = dataset[split][j][key]

            batch = collate_fn_partial(_batch)

            for func in batch_eval_funcs:
                for k, v in func(batch).items():
                    batch_metrics[k].extend(v)

        # take average on (possibly) varying number of elements (due to filtering None)
        scores = {}
        for k, v in batch_metrics.items():
            scores[k] = sum(v) / len(v)

        scores["validity"] = validity
        scores = {k: float(v) for k, v in scores.items()}

        scores = {
            "seed": seed,
            "pkl_path": pickle_path,
            "scores": scores,
        }
        scores_all[split].append(scores)

    # Save scores_all as yaml
    if not use_generated_samples:
        scores_tmp_path = os.path.join(
            args.save_score_dir, f"{split}_with_no_anno.yaml"
        )
        save_paths = [scores_tmp_path]
        output_score = scores_all

        # Create log for pasting to google spread sheet.
        log_parts = ["=== metrics ===\n"]
        _split = list(scores_all.keys())[0]
        log_parts.extend([f"{k}\n" for k in scores_all[_split][0]["scores"].keys()])
        log_parts.append("\n\n\n")
        for k, v in scores_all[_split][0]["scores"].items():
            log_parts.append(f"{v}\n")
        log = "".join(log_parts)

        for save_log_path in save_paths:
            save_log_path = save_log_path.replace(".yaml", ".txt")
            with fs.open(save_log_path, "w") as file_obj:
                file_obj.writelines(log)

    else:
        # Define save paths
        scores_all_path = os.path.join(args.input_dir, "scores_all.yaml")
        save_paths = [scores_all_path]
        try:
            g = args.input_dir.split("/")
            expid = g[5]
            expdir = g[6]
            scores_all_tmp_path = os.path.join(
                args.save_score_dir, f"{expid}___{expdir}___{ckpt_name}.yaml"
            )
            save_paths.append(scores_all_tmp_path)
        except Exception:
            pass

        scores_avg = compute_average(scores_all)
        output_score = {
            **scores_all,
            "average": scores_avg,
        }

        # Create log for pasting to google spread sheet.
        log_parts = ["=== metrics ===\n"]
        log_parts.extend(
            [f"{k}\n" for k in scores_avg[list(scores_avg.keys())[0]].keys()]
        )
        log_parts.append("\n\n\n")
        for k, v in scores_avg.items():
            log_parts.append(f"=== average {k} ===\n")
            log_parts.extend([f"{vv}\n" for kk, vv in v.items()])
            log_parts.append("\n\n\n")
        log = "".join(log_parts)

        for save_log_path in save_paths:
            save_log_path = save_log_path.replace(".yaml", ".txt")
            with fs.open(save_log_path, "w") as file_obj:
                file_obj.writelines(log)

    for save_path in save_paths:
        logger.info(f"Save score to: {save_path}")
        with fsspec.open(save_path, "w") as file_obj:
            yaml.dump(output_score, file_obj)


if __name__ == "__main__":
    main()
