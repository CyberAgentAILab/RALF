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
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torchvision.utils as vutils
from datasets.features.features import Features
from einops import rearrange
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor

from .config import init_test_config_store
from .data import collate_fn, get_dataset
from .helpers.layout_tokenizer import init_layout_tokenizer
from .helpers.retrieval_cross_dataset_wrapper import RetrievalCrossDatasetWrapper
from .helpers.retrieval_dataset_wrapper import RetrievalDatasetWrapper
from .helpers.rich_utils import CONSOLE, get_progress
from .helpers.task import get_condition
from .helpers.util import set_seed
from .helpers.visualizer import render
from .inference import (
    _validate_outputs,
    build_network,
    find_checkpoints,
    load_train_cfg,
    render_batch,
)
from .models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)

# from torchvision.io import ImageReadMode, read_image


logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra
ds.disable_caching()


cs = init_test_config_store()


def read_image(path):
    img_pil = Image.open(path).convert("RGB").resize((513, 750), Image.LANCZOS)
    img = tvF.to_tensor(img_pil)
    img = img.unsqueeze(0)
    return img


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
    # Override dataset information
    training_dataset_name = train_cfg.dataset.name
    training_data_dir = train_cfg.dataset.data_dir
    logger.info(f"{training_dataset_name=}, {test_cfg.no_anno_dataset_name=}")
    dataset_cfg = copy.deepcopy(train_cfg.dataset)
    dataset_cfg.data_dir = test_cfg.dataset_path
    dataset_cfg.name = test_cfg.no_anno_dataset_name
    max_seq_length = train_cfg.dataset.max_seq_length
    dataset, features = get_dataset(
        dataset_cfg=dataset_cfg,
        transforms=list(train_cfg.data.transforms),
    )

    use_cross_dataset = training_dataset_name[:3] != test_cfg.no_anno_dataset_name[:3]
    if use_cross_dataset:
        logger.info(f"Use cross dataset")
        dataset_cfg.name = training_dataset_name
        dataset_cfg.data_dir = training_data_dir
        training_dataset, _ = get_dataset(
            dataset_cfg=dataset_cfg,
            transforms=list(train_cfg.data.transforms),
        )
        assert len(dataset["train"]) != len(training_dataset["train"])

    if training_dataset_name == "pku" and features["label"].feature.num_classes != 3:
        features["label"].feature.num_classes = 3
    elif training_dataset_name == "cgl" and features["label"].feature.num_classes != 4:
        features["label"].feature.num_classes = 4

    use_retrieval_augment = (
        "RetrievalAugmented" in train_cfg.generator._target_
        or train_cfg.generator._target_.endswith("Retriever")
    )

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

    assert (
        test_cfg.test_split == "with_no_annotation" and test_cfg.cond_type == "uncond"
    ), ValueError(f"{test_cfg.test_split=} {test_cfg.cond_type=}")

    loaders = {}
    for split in [test_cfg.test_split]:

        if use_retrieval_augment:
            if use_cross_dataset:
                dataset[split] = RetrievalCrossDatasetWrapper(
                    dataset_source_name=test_cfg.no_anno_dataset_name,
                    dataset_reference_name=training_dataset_name,
                    dataset=dataset[split],
                    db_dataset=training_dataset["train"],
                    split=split,
                    top_k=16,
                    max_seq_length=max_seq_length,
                    retrieval_backbone=train_cfg.generator.retrieval_backbone,
                )
            else:
                dataset[split] = RetrievalDatasetWrapper(
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
            num_workers=8 if not test_cfg.debug else 0,
            batch_size=test_cfg.batch_size,
            pin_memory=False,
            collate_fn=collate_fn_partial,
            persistent_workers=False,
            drop_last=False,
        )

    # Build model
    if train_cfg.generator._target_.endswith("Retriever"):
        train_cfg.generator.retrieval_backbone = test_cfg.generator.retrieval_backbone

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

    is_parametric_model: bool = len(list(model.parameters())) > 0
    if is_parametric_model:
        ckpt_paths = find_checkpoints(
            ckpt_dir, filter_substring=test_cfg.ckpt_filter_substring
        )
        ckpt_paths = [c for c in ckpt_paths if "final" in c]
        logger.info(f"Found {ckpt_paths=}")
    else:
        ckpt_paths = [None]

    # Check dynamic topk
    if use_retrieval_augment:
        trained_k = model.top_k
        if test_cfg.dynamic_topk is not None:
            topk_cancidates = [test_cfg.dynamic_topk]
        elif "RetrievalAugmented" in train_cfg.generator._target_:
            topk_cancidates = [16]
        else:
            topk_cancidates = [train_cfg.generator.top_k]
    else:
        topk_cancidates = [None]

    logger.info(f"Use {topk_cancidates=} retrieval candidates.")

    # check model-specific sampling configs
    test_cfg.sampling = model.aggregate_sampling_config(
        sampling_cfg=test_cfg.sampling, test_cfg=test_cfg
    )

    # Create result directory
    key = f"no_anno_data_{test_cfg.cond_type}_"
    key += "_".join([f"{k}_{v}" for (k, v) in test_cfg.sampling.items()])
    if test_cfg.debug:
        key += "_debug"
    if test_cfg.debug_num_samples > 0:
        key += f"_only_{test_cfg.debug_num_samples}_samples"
    key += f"_{test_cfg.no_anno_dataset_name}"

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

        combinations = list(
            itertools.product(
                # ["test", "val"],
                # ["test"],
                [test_cfg.test_split],
                list(range(test_cfg.num_seeds)),
                topk_cancidates,
            )
        )
        logger.info(f"Start sampling with {combinations=}")
        for split, seed, _topk in combinations:

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

            if test_cfg.save_vis:
                __dirname += "_save_vis"

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
                loaders[split],
                f"{split=}, {seed=}, {_topk=}",
                # run_on_local=train_cfg.run_on_local and not test_cfg.debug,
                run_on_local=False,
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

                if test_cfg.repeat_retrieved_layouts:
                    for k, v in cond.retrieved.items():
                        if k == "index":
                            top1 = v[0]
                            v = [top1] * 16
                        else:
                            top1 = v[:, 0:1]
                            v[:, :] = top1
                        cond.retrieved[k] = v
                    model.top_k = 16

                t_start = time.time()
                outputs, _violation = model.sample(
                    batch_size=sampling_batch_size,
                    cond=cond,
                    sampling_cfg=test_cfg.sampling,
                    cond_type=test_cfg.cond_type,
                    return_violation=True,
                    use_backtrack=False,
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
                if j == 0 and not test_cfg.save_vis:
                    outputs["image"] = input_img
                    vis_layout = render_batch(outputs, features)
                    out_path = os.path.join(result_dir, f"layout_{split}_{seed}.png")
                    vutils.save_image(
                        vis_layout, out_path, normalize=False, pad_value=1.0
                    )

                torch.cuda.empty_cache()

                if test_cfg.debug:
                    break

            if test_cfg.save_vis:
                continue

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
