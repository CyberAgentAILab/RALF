import copy
import logging
import os
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
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor

from .config import init_test_config_store
from .data import collate_fn, get_dataset
from .helpers.layout_tokenizer import init_layout_tokenizer
from .helpers.retrieval_dataset_wrapper import RetrievalDatasetWrapper
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


def read_image(path):
    img_pil = Image.open(path).convert("RGB").resize((513, 750), Image.LANCZOS)
    img = tvF.to_tensor(img_pil)
    img = img.unsqueeze(0)
    return img


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
    # else:
    #     train_cfg, ckpt_dirs = _enumerate_meta(test_cfg)

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


def _enumerate_meta(
    test_cfg: DictConfig,
) -> tuple[DictConfig, list[str]]:
    # multi-seed experiment, assume seed is 0, 1, 2, ...
    fs, _ = url_to_fs(test_cfg.job_dir)

    ckpt_dirs = []
    seed = 0
    while True:
        tmp_job_dir = os.path.join(test_cfg.job_dir, str(seed))
        config_path = os.path.join(tmp_job_dir, "config.yaml")
        if fs.exists(config_path):
            if seed == 0:
                with fs.open(config_path, "rb") as file_obj:
                    train_cfg = OmegaConf.load(file_obj)
            ckpt_dirs.append(tmp_job_dir)
        else:
            break
        seed += 1
    return train_cfg, ckpt_dirs


cs = init_test_config_store()


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

    if "pku" in test_cfg.dataset_path:
        cache_table_path = "cache/dataset/pku_DATAID_TO_IDX.pt"
    elif "cgl" in test_cfg.dataset_path:
        cache_table_path = "cache/dataset/cgl_DATAID_TO_IDX.pt"

    TABLE_DATAID_TO_IDX = torch.load(cache_table_path)[test_cfg.test_split]

    if test_cfg.sample_id is None or test_cfg.sample_id == "None":
        DATA_ID = "O1CN010wAX8U1i1ilTbsxmg_!!6000000004353-0-yinhe"  # for example
        SAMPLE_IDX = TABLE_DATAID_TO_IDX[DATA_ID]
    else:
        TABLE_IDX_to_DATAID = {v: k for k, v in TABLE_DATAID_TO_IDX.items()}
        SAMPLE_IDX = test_cfg.sample_id
        DATA_ID = TABLE_IDX_to_DATAID[SAMPLE_IDX]

    max_seq_length = train_cfg.dataset.max_seq_length
    dataset, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
    )

    use_retrieval_augment = (
        "RetrievalAugmented" in train_cfg.generator._target_
        or train_cfg.generator._target_.endswith("Retriever")
    )
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

    if use_retrieval_augment:

        dataset[test_cfg.test_split] = RetrievalDatasetWrapper(
            dataset_name=train_cfg.dataset.name,
            dataset=dataset[test_cfg.test_split],
            db_dataset=dataset["train"],
            split=test_cfg.test_split,
            top_k=16,
            max_seq_length=max_seq_length,
            retrieval_backbone=train_cfg.generator.retrieval_backbone,
            random_retrieval=random_retrieval,
            saliency_k=None,
            inference_num_saliency=None,
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
            _topk = train_cfg.generator.top_k
        else:
            _topk = train_cfg.generator.top_k
    else:
        _topk = "None"

    # check model-specific sampling configs
    test_cfg.sampling = model.aggregate_sampling_config(
        sampling_cfg=test_cfg.sampling, test_cfg=test_cfg
    )

    # Create result directory
    key = os.path.join("single_data", f"{test_cfg.cond_type}_")
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
            _use_backtrack = True
        else:
            _use_backtrack = False

        # Create result directory
        if train_cfg.generator._target_.endswith("Retriever"):
            __dirname = f"{_dirname}_{model.retrieval_backbone}"
        elif "RetrievalAugmented" in train_cfg.generator._target_:
            logger.info(f"Dynamic topk is enabled, update {trained_k=} to {_topk=}")
            model.top_k = _topk
            __dirname = f"{_dirname}_topk_{_topk}"
            if random_retrieval:
                __dirname += "_randomdb"
        else:
            __dirname = _dirname

        if _use_backtrack:
            __dirname += "_backtrack"

        # Create result directory
        result_dir = os.path.join(
            test_cfg.result_dir, __dirname, f"{test_cfg.test_split}_{str(DATA_ID)}"
        )
        fs_result, _ = url_to_fs(result_dir)
        if not fs_result.exists(result_dir):
            fs_result.mkdir(result_dir)
        logger.info(f"Results saved to {result_dir=}")

        data = dataset[test_cfg.test_split][SAMPLE_IDX]
        _input_id = data["id"]
        batch = collate_fn_partial([data])
        tokenizer = kwargs.get("tokenizer", None)

        batch_ori = copy.deepcopy(batch)

        if model.auxilary_task != "refinement":
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
            else:
                cond = {
                    k: v.to(device) for k, v in cond.items() if isinstance(v, Tensor)
                }
                sampling_batch_size = cond["image"].size(0)
                if use_retrieval_augment:
                    cond["retrieved"] = batch["retrieved"]

        for seed in list(range(test_cfg.num_seeds)):

            if model.auxilary_task == "refinement":
                cond, batch = get_condition(
                    batch=copy.deepcopy(batch_ori),
                    cond_type=test_cfg.cond_type,
                    tokenizer=tokenizer,
                    model_type=type(model).__name__,
                )  # Deeply copy batch to avoid moving the reference of batch to GPU.
                if isinstance(cond, ConditionalInputsForDiscreteLayout) or isinstance(
                    cond, RetrievalAugmentedConditionalInputsForDiscreteLayout
                ):
                    cond = cond.to(device)
                    sampling_batch_size = cond.image.size(0)
                else:
                    cond = {
                        k: v.to(device)
                        for k, v in cond.items()
                        if isinstance(v, Tensor)
                    }
                    sampling_batch_size = cond["image"].size(0)
                    if use_retrieval_augment:
                        cond["retrieved"] = batch["retrieved"]

                # Save GT
                input_canvas = render(
                    prediction=batch,
                    label_feature=model.features["label"].feature,
                )
                save_input_path = os.path.join(
                    result_dir, f"{DATA_ID}_seed{seed}_input.png"
                )
                vutils.save_image(
                    input_canvas, save_input_path, normalize=False, pad_value=1.0
                )

            save_path = os.path.join(result_dir, f"{DATA_ID}_seed{seed}.png")
            if fs_result.exists(save_path) and not test_cfg.debug:
                logger.info(f"Skip {save_path}, already exists.")
                continue

            logger.info(f"DATA_ID: {DATA_ID}, seed: {seed} / {test_cfg.num_seeds}")

            set_seed(seed)

            outputs = model.sample(
                batch_size=sampling_batch_size,
                cond=copy.deepcopy(cond),
                sampling_cfg=test_cfg.sampling,
                cond_type=test_cfg.cond_type,
                return_violation=False,
                use_backtrack=_use_backtrack,
                return_decoded_cond=True,
            )
            outputs["image"] = batch_ori["image"].clone()
            pred_layout = render(
                prediction=outputs,
                label_feature=model.features["label"].feature,
            )
            vutils.save_image(pred_layout, save_path, normalize=False, pad_value=1.0)
            logger.info(f"Save to {save_path}")

            torch.cuda.empty_cache()

            if test_cfg.cond_type == "relation" or seed == 0:
                # save the first seed for debugging
                task = model.auxilary_task
                save_task_token_path = os.path.join(
                    result_dir, f"{DATA_ID}_cond_{task}_seed{seed}.txt"
                )
                if test_cfg.cond_type == "relation":
                    outputs["decoded_tokens"] = [
                        str(s) for s in outputs["decoded_tokens"][0]
                    ]
                with open(save_task_token_path, "w") as f:
                    t = " | ".join(outputs["decoded_tokens"][0])
                    f.write(t)

            if test_cfg.debug:
                break

        if test_cfg.debug:
            break


if __name__ == "__main__":
    main()
