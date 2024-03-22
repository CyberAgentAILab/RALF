import argparse
import itertools
import logging
import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Union

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from image2layout.train.data import collate_fn, get_dataset
from image2layout.train.fid.model import FIDNetV3, load_fidnet_v3
from image2layout.train.global_variables import GEO_KEYS
from image2layout.train.helpers.metric import (
    SingletonTimmInceptionV3,
    compute_alignment,
    compute_generative_model_scores,
    compute_overlap,
    compute_overlay,
    compute_rshm,
    compute_saliency_aware_metrics,
    compute_underlay_effectiveness,
    compute_validity,
)
from image2layout.train.helpers.rich_utils import CONSOLE, get_progress
from image2layout.train.helpers.task import REFINEMENT_NOISE_STD
from image2layout.train.helpers.util import set_seed, box_cxcywh_to_xyxy
from image2layout.train.helpers.visualizer import mask_out_bbox_area
from omegaconf import OmegaConf
from torch import Tensor

logger = logging.getLogger(__name__)


KEYS = ["label", "width", "height", "center_x", "center_y", "id"]


def perturb_layout(generated_samples):
    outputs = []
    logger.info(f"Add noise to layout with std={REFINEMENT_NOISE_STD}")
    for batch in generated_samples:

        for key in GEO_KEYS:

            noise = torch.normal(
                0,
                REFINEMENT_NOISE_STD,
                size=(len(batch[key]),),
            )
            batch[key] = torch.tensor(batch[key])
            batch[key] = torch.clamp(batch[key] + noise, min=0.0, max=1.0)
            batch[key] = batch[key].tolist()

        outputs.append(batch)

    return outputs


def load_pkl(pickle_path: str) -> tuple:
    """
    Pickle is saved using only python's standard modules for potability.
    DictConfig instances are saved as dicts, so we convert it back for convenience.
    """

    fs, _ = fsspec.core.url_to_fs(pickle_path)
    assert fs.exists(pickle_path), f"{pickle_path} not found"
    logger.info(f"Load pickle from {pickle_path}")
    with fs.open(pickle_path, "rb") as file_obj:
        data = pickle.load(file_obj)

    base = pickle_path.split("/")[-1].replace(".pkl", "").split("_")
    split = base[0]
    seed = base[1]
    ckpt_name: str = pickle_path.split("/")[-2].split("_")[-1]

    return (
        fs,
        data["results"],
        OmegaConf.create(data["train_cfg"]),
        OmegaConf.create(data["test_cfg"]),
        split,
        seed,
        ckpt_name,
    )


def print_scores(scores: dict[str, list[float]]) -> None:
    tex_text = ""
    for k, v in scores.items():
        mean, std = np.mean(v), np.std(v)
        stdp = std * 100.0 / mean
        CONSOLE.print(f"\t{k}: {mean:.4f} ({stdp:.4f}%)")
        tex_text += f"& {mean:.4f}\\std{{{stdp:.1f}}}\% "

    CONSOLE.print(tex_text + "\\\\")


def compute_average(
    scores_all: dict[str, list[dict[str, Union[float, dict[str, float]]]]]
) -> dict[str, dict[str, float]]:
    scores_avg = {k: defaultdict(list) for k in scores_all.keys()}

    for split, scores in scores_all.items():
        for score in scores:
            for k, v in score["scores"].items():
                # Single check for type
                if isinstance(v, float):
                    scores_avg[split][k].append(v)
                else:  # if it's not a float, then it's assumed to be a dict based on the provided type hints.
                    for kk, vv in v.items():
                        scores_avg[split][f"{k}_{kk}"].append(vv)

        # Compute the average
        for key, values in scores_avg[split].items():
            scores_avg[split][key] = sum(values) / len(values)

    # convert back to standard dict
    return {k: dict(v) for k, v in scores_avg.items()}


@torch.no_grad()  # type: ignore
def _extract_layout_feautures(
    loaders: dict[str, torch.utils.data.DataLoader],
    fid_model_layout: FIDNetV3,
    fid_model_inceptionv3: SingletonTimmInceptionV3,
    device: torch.device,
    run_on_local: bool,
) -> dict[str, Tensor]:
    feats_gts = {
        "layout": {},
        "image": {},
    }
    _feats_gts_layout = defaultdict(list)
    _feats_gts_image = defaultdict(list)
    for split in ["val", "test"]:
        pbar = get_progress(
            loaders[split], f"[{split}] Computing gt features for FID", run_on_local
        )
        for batch in pbar:
            batch = {
                k: v.to(device)
                for (k, v) in batch.items()
                if k
                not in [
                    "saliency",
                    "id",
                ]
            }
            # Extract layout feature
            _feat_layout = fid_model_layout.extract_features(batch)
            _feats_gts_layout[split].append(_feat_layout.detach().cpu())

            # Extract image feature
            # 1. Apply layout-based mask to image
            center_x = batch["center_x"]  # [bs, max_elem]
            center_y = batch["center_y"]  # [bs, max_elem]
            width = batch["width"]  # [bs, max_elem]
            height = batch["height"]  # [bs, max_elem]
            bbox_cxcywh = torch.stack(
                [center_x, center_y, width, height], dim=-1
            )  # [bs, max_elem, 4]
            bbox_xyxy = box_cxcywh_to_xyxy(bbox_cxcywh)  # [bs, max_elem, 4]
            image_maskout = mask_out_bbox_area(batch["image"], bbox_xyxy)

            # 2. Extract image feature
            _feat_image = fid_model_inceptionv3(image_maskout)  # [bs, 2048]
            _feats_gts_image[split].append(_feat_image.detach().cpu())

        # Layout feature
        _feats_gts_layout[split] = torch.cat(_feats_gts_layout[split], dim=0)
        feats_gts["layout"][split] = _feats_gts_layout[split]

        # image feature
        _feats_gts_image[split] = torch.cat(_feats_gts_image[split], dim=0)
        feats_gts["image"][split] = _feats_gts_image[split]

    return feats_gts


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--fid-weight-dir",
        type=str,
        default=None,
    )
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
        "--add-noise",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--run-on-local",
        action="store_true",
    )
    parser.add_argument("--batch-size", type=int, default=32)
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
        if fs.exists(scores_all_path):
            logger.info(f"Find {scores_all_path}. Finish!")
            return None
        pickle_paths = fs.glob(os.path.join(args.input_dir, "*.pkl"))
        logger.info(f"Found pickle files: {pickle_paths=}")
        calculate_paired_score = True
    else:
        pickle_paths = [None]
        ckpt_name = "ground-truth dataset"
        seed = "None"
        split = args.load_gt_split
        calculate_paired_score = split != "train"

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
                "batch_size": 128,
                "dataset_path": args.dataset_path,
            }
        )
        logger.info(f"Use ground-truth {split=} dataset")

    # Build dataset
    if use_generated_samples:
        train_cfg, test_cfg = load_pkl(pickle_paths[0])[2:4]
    dataset, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
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
    for _split in ["val", "test"]:
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

    # Load FID models for layout
    dataset_name = args.dataset_path.split("/")[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fid_model_layout = FIDNetV3(
        num_label=feature_label.num_classes,
        max_bbox=train_cfg.dataset.max_seq_length,
    )
    assert args.fid_weight_dir is not None
    fid_model_layout = load_fidnet_v3(fid_model_layout, args.fid_weight_dir).to(device)

    # Load FID models for image
    fid_model_inceptionv3 = SingletonTimmInceptionV3()

    # Compute features of ground-truth data
    if not os.path.exists("cache/eval_gt_features"):
        os.makedirs("cache/eval_gt_features")
    cache_path = f"cache/eval_gt_features/{dataset_name}_FIDNetV3_features.pth"
    if os.path.exists(cache_path):
        logger.info(f"Find the cache in {cache_path} and loading ...")
        feats_gts: dict[str, dict[str, Tensor]] = torch.load(cache_path)
    else:
        logger.info(f"Extract layout feat. to {cache_path=}")
        feats_gts = _extract_layout_feautures(
            loaders,
            fid_model_layout,
            fid_model_inceptionv3,
            device,
            train_cfg.run_on_local,
        )
        torch.save(feats_gts, cache_path)

    scores_all = defaultdict(list)
    for pickle_path in pickle_paths:
        # Load picjke
        if use_generated_samples:
            (
                fs,
                generated_samples,
                train_cfg,
                test_cfg,
                split,
                seed,
                ckpt_name,
            ) = load_pkl(pickle_path)
        else:
            # Load ground truth samples for gt-gt evaluation
            generated_samples = [
                {k: v for k, v in dataset[split][i].items() if k in KEYS}
                for i in range(len(dataset[split]))
            ]

        generated_samples, validity = compute_validity(generated_samples)

        if not use_generated_samples and args.add_noise:
            generated_samples = perturb_layout(generated_samples)

        # Attach image and saliency to generated samples.
        assert len(dataset[split]) == len(generated_samples)

        # compute scores for each run
        logger.info("Evaluation start!!")

        feats_preds: dict[str, list[Tensor]] = {
            "layout": [],
            "image": [],
        }
        batch_metrics = defaultdict(list)

        # Compute metrics and extract features.
        pbar = get_progress(
            range(0, len(generated_samples), batch_size),
            "Eval generated samples",
            run_on_local=args.run_on_local,
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
            batch_gpu = {k: v.to(device) for (k, v) in batch.items() if k != "id"}

            for func in batch_eval_funcs:
                for k, v in func(batch).items():
                    batch_metrics[k].extend(v)

            if calculate_paired_score:
                # Extract layout feature
                with torch.no_grad():
                    _feat: Tensor = fid_model_layout.extract_features(
                        batch_gpu
                    )  # [bs, 256]
                feats_preds["layout"].append(_feat.detach().cpu())

                # Extract image feature
                # 1. Apply layout-based mask to image
                center_x = batch_gpu["center_x"]  # [bs, max_elem]
                center_y = batch_gpu["center_y"]  # [bs, max_elem]
                width = batch_gpu["width"]  # [bs, max_elem]
                height = batch_gpu["height"]  # [bs, max_elem]

                bbox_cxcywh = torch.stack(
                    [center_x, center_y, width, height], dim=-1
                )  # [bs, max_elem, 4]
                bbox_xyxy = box_cxcywh_to_xyxy(bbox_cxcywh)  # [bs, max_elem, 4]
                image_maskout = mask_out_bbox_area(batch_gpu["image"], bbox_xyxy)
                with torch.no_grad():
                    _feat_image = fid_model_inceptionv3(image_maskout)  # [bs, 2048]
                feats_preds["image"].append(_feat_image.detach().cpu())

        scores = {}
        for k, v in batch_metrics.items():
            scores[k] = sum(v) / len(v)

        scores["validity"] = validity
        scores = {k: float(v) for k, v in scores.items()}

        if calculate_paired_score:
            feats_preds["layout"]: torch.Tensor = torch.cat(feats_preds["layout"], dim=0)  # type: ignore
            feats_preds["image"]: torch.Tensor = torch.cat(feats_preds["image"], dim=0)  # type: ignore
            if use_generated_samples:
                target_splits = [split]
            else:
                target_splits = ["val", "test"]
            combinations = list(itertools.product(target_splits, ["layout"]))
            for target_split, modality in combinations:
                logger.info(f"Compute FID for {target_split}--{modality} features")
                paired_score: dict[str, float] = {}
                assert len(feats_gts[modality][target_split]) == len(
                    feats_preds[modality]
                ), f"GT {len(feats_gts[modality][target_split])} != Pred {len(feats_preds[modality])}"
                # Compute FID for layout
                _score = compute_generative_model_scores(
                    feats_gts[modality][target_split], feats_preds[modality]
                )
                # Update key of scores
                _score = {
                    f"{target_split}_{k}_{modality}": v for k, v in _score.items()
                }
                _mse = F.mse_loss(
                    feats_gts[modality][target_split], feats_preds[modality]
                ).item()
                _name = "pred" if use_generated_samples else "gt"
                logger.info(
                    f"[{modality}] MSE between gt ({target_split}) and {_name} ({split}) features: {_mse}"
                )
                scores = {**scores, **_score}

        scores = {
            "seed": seed,
            "pkl_path": pickle_path,
            "scores": scores,
        }
        scores_all[split].append(scores)

    # Save scores_all as yaml
    if not use_generated_samples:
        scores_tmp_path = os.path.join(args.save_score_dir, f"{split}_set.yaml")
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
