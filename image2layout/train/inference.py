import copy
import itertools
import logging
import os
import pickle
import time
from functools import partial
from typing import Optional

import datasets as ds
import hydra
import torch
import torch.nn as nn
import torchvision.utils as vutils
from datasets.features.features import Features
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .config import init_test_config_store
from .data import collate_fn, get_dataset
from .helpers.layout_tokenizer import init_layout_tokenizer
from .helpers.random_retrieval_dataset_wrapper import RandomRetrievalDatasetWrapper
from .helpers.retrieval_dataset_wrapper import RetrievalDatasetWrapper
from .helpers.rich_utils import CONSOLE, get_progress
from .helpers.task import get_condition
from .helpers.util import set_seed
from .helpers.visualizer import render
from .models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)

logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra
ds.disable_caching()

cs = init_test_config_store()


def load_train_cfg(job_dir: str) -> tuple[LocalFileSystem, DictConfig, list[str]]:

    fs, _ = url_to_fs(job_dir)
    if not fs.exists(job_dir):
        raise FileNotFoundError(f"{job_dir} not found")

    config_path = os.path.join(job_dir, "config.yaml")
    assert fs.exists(config_path), f"{config_path} not found"

    if fs.exists(config_path):
        with fs.open(config_path, "rb") as file_obj:
            train_cfg = OmegaConf.load(file_obj)
        ckpt_dirs = [job_dir]
    else:
        raise ValueError("config.yaml not found")

    return (fs, train_cfg, ckpt_dirs)


def find_checkpoints(
    ckpt_dir: str, filter_substring: Optional[str] = None
) -> list[str]:
    fs, path_prefix = url_to_fs(ckpt_dir)
    ckpt_paths: list[str] = fs.glob(os.path.join(path_prefix, "*pt"))  # type: ignore
    if filter_substring:
        logger.info(f"Filter checkpoints by {filter_substring=}")
        ckpt_paths = [p for p in ckpt_paths if filter_substring in p]
    else:
        logger.info(f"Find {len(ckpt_paths)} checkpoints in {path_prefix}")
    return ckpt_paths


def build_network(
    train_cfg: DictConfig,
    # ckpt_dir: str,
    # best_or_final: str,
    features: Features,
    max_seq_length: int,
    db_dataset: Optional[torch.utils.data.Dataset] = None,
) -> tuple[nn.Module, dict]:
    """
    note: db_dataset is necessary for retrieval-based models.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"features": features}
    if train_cfg.data.tokenization:
        kwargs["tokenizer"] = init_layout_tokenizer(
            tokenizer_cfg=train_cfg.tokenizer,
            dataset_cfg=train_cfg.dataset,
            label_feature=features["label"].feature,
        )
        use_sorted_seq = "shuffle" not in train_cfg.data.transforms
        kwargs["tokenizer"].use_sorted_seq = use_sorted_seq

    if db_dataset is not None:
        kwargs["db_dataset"] = db_dataset
        kwargs["max_seq_length"] = max_seq_length
        kwargs["dataset_name"] = train_cfg.dataset.name
    if train_cfg.generator._target_.endswith("Retriever"):
        kwargs["top_k"] = 1
        kwargs["retrieval_backbone"] = train_cfg.generator.retrieval_backbone
    model = instantiate(train_cfg.generator)(**kwargs)
    model.eval()
    model = model.to(device)

    return model, kwargs


def _validate_outputs(layouts: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    keys = set(["label", "mask", "center_x", "center_y", "width", "height", "id"])
    assert set(layouts.keys()) == keys

    outputs = []
    for b in range(layouts["mask"].size(0)):
        mask = layouts["mask"][b]
        output = {}
        for key in layouts:
            if key == "mask":
                continue
            elif key == "id":
                output["id"] = layouts[key][b]
            else:
                # append only mask is True
                output[key] = layouts[key][b][mask].tolist()
        outputs.append(output)
    return outputs


def render_batch(input, features):
    pred_layout_image = []
    for idx in range(input["label"].size(0)):
        _batch = {k: v[idx] for k, v in input.items() if isinstance(v, Tensor)}
        _layout_image = render(
            prediction=_batch,
            label_feature=features["label"].feature,
        )  # [3, H, W]
        pred_layout_image.append(_layout_image)
    pred_layout_image = torch.stack(pred_layout_image)
    return pred_layout_image


def render_input_output(input, output, features, input_img):

    img_input = render_batch(input, features)

    # output
    output["image"] = input_img
    img_output = render_batch(output, features)

    # concat
    output = torch.cat([img_input, img_output], dim=2)

    return output


@hydra.main(version_base="1.2", config_name="test_config")
def main(test_cfg: DictConfig) -> None:
    logger.info(test_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"{test_cfg.job_dir=}")
    fs_ckpt, train_cfg, ckpt_dirs = load_train_cfg(test_cfg.job_dir)
    assert len(ckpt_dirs) == 1
    ckpt_dir = ckpt_dirs[0]
    logger.info(f"Found {ckpt_dir=}")

    # Load dataset
    train_cfg.dataset.data_dir = test_cfg.dataset_path
    max_seq_length = train_cfg.dataset.max_seq_length
    dataset, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
    )

    use_retrieval_augment = (
        "RetrievalAugmented" in train_cfg.generator._target_
        or train_cfg.generator._target_.endswith("Retriever")
    )
    is_copy_generator = train_cfg.generator._target_.endswith("Retriever")
    # Build model
    if train_cfg.generator._target_.endswith("Retriever"):
        train_cfg.generator.retrieval_backbone = test_cfg.generator.retrieval_backbone

    # Load dataloader
    if max_seq_length < 0:
        max_seq_length = None
    collate_fn_partial = partial(collate_fn, max_seq_length=max_seq_length)

    if "RetrievalAugmented" in train_cfg.generator._target_:
        random_retrieval = train_cfg.generator.random_retrieval
        if test_cfg.use_db_dataset:
            random_retrieval = False
            logger.info("use_db_dataset is enabled, disable random_retrieval")
    else:
        random_retrieval = False

    if test_cfg.test_split == "val" and test_cfg.cond_type == "relation":
        return

    loaders = {}
    for split in [test_cfg.test_split]:

        if use_retrieval_augment:

            if random_retrieval:
                DatasetWrapperClass = RandomRetrievalDatasetWrapper
            else:
                DatasetWrapperClass = RetrievalDatasetWrapper
            logger.info(f"{DatasetWrapperClass=} with {random_retrieval=}")

            dataset[split] = DatasetWrapperClass(
                dataset_name=train_cfg.dataset.name,
                dataset=dataset[split],
                db_dataset=dataset["train"],
                split=split,
                top_k=16,
                max_seq_length=max_seq_length,
                retrieval_backbone=train_cfg.generator.retrieval_backbone,
                random_retrieval=random_retrieval,
                saliency_k=None,
                inference_num_saliency=None,
            )

        loaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            num_workers=2 if not test_cfg.debug else 0,
            batch_size=test_cfg.batch_size,
            pin_memory=False,
            collate_fn=collate_fn_partial,
            persistent_workers=False,
            drop_last=False,
        )

    model, kwargs = build_network(
        train_cfg,
        features,
        max_seq_length,
        dataset["train"] if use_retrieval_augment else None,
    )

    if "AuxilaryTask" in type(model).__name__:
        if not model.use_multitask:
            if model.auxilary_task != test_cfg.cond_type:
                logger.info(f"Skip, {model.auxilary_task=} != {test_cfg.cond_type=}")
                return None
    elif "CGLGenerator" == type(model).__name__:
        if model.auxilary_task != test_cfg.cond_type:
            logger.info(
                f"[CGLGenerator] Skip, {model.auxilary_task=} != {test_cfg.cond_type=}"
            )
            return None

    # Cache features
    if test_cfg.preload_data:
        loaders_cache = {}
        for split in [test_cfg.test_split]:
            loaders_cache[split] = []
            pbar_cache_loader = get_progress(
                loaders[split],
                f"Cache {split} dataloader",
                run_on_local=train_cfg.run_on_local and not test_cfg.debug,
            )
            for batch in pbar_cache_loader:
                loaders_cache[split].append(batch)

                if test_cfg.debug:
                    break
            if test_cfg.debug:
                break
    else:
        loaders_cache = loaders

    is_parametric_model: bool = len(list(model.parameters())) > 0
    if is_parametric_model:
        ckpt_paths = find_checkpoints(
            ckpt_dir, filter_substring=test_cfg.ckpt_filter_substring
        )
        if test_cfg.cond_type == "relation":
            ckpt_paths = [c for c in ckpt_paths if "final" in c]
        else:
            ckpt_paths = [c for c in ckpt_paths if "final" in c or "epoch" in c]
        logger.info(f"Found {ckpt_paths=}")
    else:
        ckpt_paths = [None]

    # Check dynamic topk
    if use_retrieval_augment:
        trained_k = model.top_k
        if "RetrievalAugmented" in train_cfg.generator._target_:
            topk_cancidates = [train_cfg.generator.top_k]
        else:
            topk_cancidates = [train_cfg.generator.top_k]
    else:
        topk_cancidates = [None]

    # check model-specific sampling configs
    test_cfg.sampling = model.aggregate_sampling_config(
        sampling_cfg=test_cfg.sampling, test_cfg=test_cfg
    )

    # Create result directory
    key = f"generated_samples_{test_cfg.cond_type}_"
    key += "_".join([f"{k}_{v}" for (k, v) in test_cfg.sampling.items()])
    if test_cfg.debug:
        key += "_debug"
    if test_cfg.debug_num_samples > 0:
        key += f"_only_{test_cfg.debug_num_samples}_samples"

    logger.info(f"dirname: {key}")

    for ckpt_path in ckpt_paths:

        # Load pre-trained weifht
        if ckpt_path is not None:
            logger.info(f"Load from {ckpt_path=}")
            with fs_ckpt.open(ckpt_path) as f:
                model.load_state_dict(torch.load(f, map_location="cpu"))
            ckpt_name = os.path.basename(ckpt_path).split("_")[1]
        else:
            logger.info("Using retrieval model, no checkpoint is loaded.")
            ckpt_name = "retrieval"
        _dirname = f"{key}_{ckpt_name}"

        if test_cfg.cond_type == "relation":
            use_backtrack = [True]
        else:
            use_backtrack = [False]

        combinations = list(
            itertools.product(
                [test_cfg.test_split],
                list(range(test_cfg.num_seeds)),
                topk_cancidates,
                use_backtrack,
            )
        )
        logger.info(f"Start sampling with {combinations=}")
        for split, seed, _topk, _use_backtrack in combinations:

            if train_cfg.generator._target_.endswith("Retriever"):
                __dirname = f"{_dirname}_{model.retrieval_backbone}"
            elif "RetrievalAugmented" in train_cfg.generator._target_:
                logger.info(f"Dynamic topk is enabled, update {trained_k=} to {_topk=}")
                model.top_k = _topk
                __dirname = f"{_dirname}_dynamictopk_{_topk}"
                if random_retrieval:
                    __dirname += "_randomdb"
            else:
                __dirname = _dirname

            if _use_backtrack:
                __dirname += "_backtrack"

            # Create result directory
            result_dir = os.path.join(test_cfg.result_dir, __dirname)
            fs_result, _ = url_to_fs(result_dir)
            if not fs_result.exists(result_dir):
                fs_result.mkdir(result_dir)
            logger.info(f"Results saved to {result_dir=}")

            pkl_file = os.path.join(result_dir, f"{split}_{seed}.pkl")
            if fs_result.exists(pkl_file) and not test_cfg.debug:
                logger.info(f"Skip {pkl_file}, already exists.")
                continue

            logger.info(f"split: {split=}, seed: {seed=} / {test_cfg.num_seeds}")

            set_seed(seed)
            t_total = 0.0
            N_total = 0
            inputs = []
            results = []

            pbar_loader = get_progress(
                loaders_cache[split],
                f"{split=}, {seed=}, {_topk=}",
                run_on_local=train_cfg.run_on_local and not test_cfg.debug,
            )

            VIOLATION = {
                "total": 0,
                "viorated": 0,
            }

            for j, batch in enumerate(pbar_loader):
                tokenizer = kwargs.get("tokenizer", None)
                cond, batch = get_condition(
                    batch=copy.deepcopy(batch),
                    cond_type=test_cfg.cond_type,
                    tokenizer=tokenizer,
                    model_type=type(model).__name__,
                )  # Deeply copy batch to avoid moving the reference of batch to GPU.
                if isinstance(cond, ConditionalInputsForDiscreteLayout) or isinstance(
                    cond, RetrievalAugmentedConditionalInputsForDiscreteLayout
                ):
                    cond = cond.to(device)
                    sampling_batch_size = cond.image.size(0)
                    input_img = cond.image
                else:
                    cond = {
                        k: v.to(device)
                        for k, v in cond.items()
                        if isinstance(v, Tensor)
                    }
                    sampling_batch_size = cond["image"].size(0)
                    input_img = cond["image"]
                    if use_retrieval_augment:
                        cond["retrieved"] = batch["retrieved"]

                t_start = time.time()
                if is_copy_generator:
                    # Top1 Retrieval
                    outputs = {
                        k: batch["retrieved"][k][:, 0]
                        for k in [
                            "label",
                            "mask",
                            "center_x",
                            "center_y",
                            "width",
                            "height",
                        ]
                    }
                else:
                    outputs, _violation = model.sample(
                        batch_size=sampling_batch_size,
                        cond=cond,
                        sampling_cfg=test_cfg.sampling,
                        cond_type=test_cfg.cond_type,
                        return_violation=True,
                        use_backtrack=_use_backtrack,
                        j=j,
                    )

                    if _violation is not None:
                        for _k in _violation.keys():
                            VIOLATION[_k] += _violation[_k]

                t_end = time.time()
                t_total += t_end - t_start
                N_total += sampling_batch_size

                outputs["id"] = batch["id"]
                results.extend(_validate_outputs(outputs))
                if j == 0:
                    # # sanity check
                    # outputs["image"] = input_img
                    # layout = render(
                    #     prediction=outputs, label_feature=features["label"].feature
                    # )
                    # out_path = os.path.join(result_dir, f"layout_{split}_{seed}.png")
                    # vutils.save_image(layout, out_path, normalize=False)
                    vis_layout = render_input_output(
                        batch, outputs, features, input_img
                    )  # Visualize input and output
                    out_path = os.path.join(result_dir, f"layout_{split}_{seed}.png")
                    vutils.save_image(
                        vis_layout, out_path, normalize=False, pad_value=1.0
                    )

                torch.cuda.empty_cache()

                if test_cfg.debug:
                    break

            dummy_cfg = copy.deepcopy(train_cfg)
            data = {
                "results": results,
                "train_cfg": dummy_cfg,
                "test_cfg": test_cfg,
            }
            if len(inputs) > 0:
                data["inputs"] = inputs

            # make sure .pkl can be loaded without reference to the original code.
            for k, v in data.items():
                if isinstance(v, DictConfig):
                    data[k] = OmegaConf.to_container(v)  # save as dict
            logger.info(f"Save pickle file to {pkl_file}.")
            with fs_result.open(pkl_file, "wb") as file_obj:
                pickle.dump(data, file_obj)

            # Save violation rate
            if VIOLATION["total"] > 0:
                vio_file = os.path.join(result_dir, f"{split}_{seed}_violation.csv")
                logger.info(f"Save violation into {vio_file}")
                with fs_result.open(vio_file, "w") as file_obj:
                    file_obj.write(
                        f"total,{VIOLATION['total']}\nviorated,{VIOLATION['viorated']}\nvioration_rate,{100 * VIOLATION['viorated'] / VIOLATION['total']}"
                    )

            CONSOLE.print(N_total)
            CONSOLE.print(f"ms per sample: {1e3 * t_total / N_total}")

            if test_cfg.debug:
                break

        if test_cfg.debug:
            break


if __name__ == "__main__":
    main()
