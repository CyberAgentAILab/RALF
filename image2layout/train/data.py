import json
import logging
import os
from typing import Optional

import datasets as ds
import fsspec
import torch
from image2layout.train.global_variables import DUMMY_LAYOUT, RETRIEVED_KEYS
from image2layout.train.helpers.hfds_instance_wise_transforms import (
    hfds_instance_wise_trasnform_factory,
)
from image2layout.train.helpers.metric import compute_validity
from image2layout.train.helpers.util import (
    dict_of_list_to_list_of_dict,
    list_of_dict_to_dict_of_list,
)
from torch import Tensor
from torch.utils.data import default_collate

from .config.dataset import BaseDatasetConfig

logger = logging.getLogger(__name__)


def find_data_files(
    data_dir: str, splits: Optional[list[str]] = None
) -> dict[str, list[str]]:
    """Find data files in each split."""
    splits = splits or ["train", "val", "test"]

    def _glob(url_prefix: str) -> list[str]:
        fs, path_prefix = fsspec.core.url_to_fs(url_prefix)
        return fs.glob(path_prefix + "*")  # type: ignore

    data_files = {split: _glob(os.path.join(data_dir, split)) for split in splits}
    if any(len(x) == 0 for x in data_files.values()):
        raise FileNotFoundError(f"No dataset file found at {data_dir} with {splits}")
    return data_files


def collate_fn(
    batch,
    max_seq_length=None,
    validity_check: Optional[compute_validity] = None,
):
    """
    Custom function to merge varying-length inputs into a single batch.
    For padding, we used the values defined in pad() function above.
    """
    assert (
        validity_check is None or validity_check == compute_validity
    ), f"validity_check function is {validity_check}"

    B = len(batch)

    # delete special column used to pass the name of transforms
    for i in range(B):
        if "transforms" in batch[i]:
            del batch[i]["transforms"]

    # check if all the elements in a batch have the length > 0
    # (sometimes, generated layouts are empty)
    total_elems = []
    for i in range(B):
        n = len(batch[i]["label"])
        if n == 0:
            # add dummy element to continue evaluation
            for k in DUMMY_LAYOUT:
                batch[i][k] = DUMMY_LAYOUT[k]
            n = 1
        total_elems.append(n)

    output = {}

    for key in batch[0].keys():

        if key == "retrieved":
            retrieved = {}
            for _rkey in RETRIEVED_KEYS:
                _rvalue = [
                    val if isinstance(val, Tensor) else val[0]
                    for val in (example["retrieved"][0][_rkey] for example in batch)
                ]
                # [B, N, ...], N means the number of retrieved samples
                retrieved[_rkey] = torch.stack(_rvalue)
            output = {**output, **retrieved}
            continue

        main_data = batch[0][key]
        if not isinstance(main_data, list) or len(main_data) == 0:
            continue

        # number of elements in a layout varies, so we need padding
        if isinstance(main_data[0], int):
            pad_value = 0
        elif isinstance(main_data[0], float):
            pad_value = 0.0
        else:
            # assume this type of data works without padding
            batch[i][key] = torch.tensor(batch[i][key])
            continue

        for i in range(B):
            batch[i][key] = torch.tensor(
                batch[i][key] + [pad_value] * (max_seq_length - total_elems[i])
            )

    for i in range(B):
        n = total_elems[i]
        batch[i]["mask"] = torch.BoolTensor([True] * n + [False] * (max_seq_length - n))

    if validity_check is not None:
        batch, _ = validity_check(batch)

    output = {**output, **default_collate(batch)}
    return output


def _composed_transform(inputs: dict) -> dict:
    """
    Assume that inputs has the following structure:
        data = {
            "image",
            "label",
            "center_x",
            "center_y",
            ...,
            "retrieved" : {
                "id",
                "center_x",
            }
        }
    In this function, we don't apply any transforms to data["retrieved"],
    because FID feature extractor ignores the order of elements, which has
    no positional embeddings.
    Thus, without shuffling orders does not affect the performance.
    """
    inputs_ld = dict_of_list_to_list_of_dict(inputs)
    for name in inputs["transforms"][0]:
        inputs_ld = [hfds_instance_wise_trasnform_factory(name)(x) for x in inputs_ld]
    inputs = list_of_dict_to_dict_of_list(inputs_ld)
    return inputs


def get_dataset(
    dataset_cfg: BaseDatasetConfig,
    transforms: Optional[list[str]] = None,
    remove_column_names: Optional[list[str]] = None,
) -> tuple[ds.DatasetDict, ds.Features]:
    if dataset_cfg.path:
        # load from huggingface datasets
        dataset = ds.load_dataset(dataset_cfg.path)
    else:
        data_dir: str = dataset_cfg.data_dir
        data_files = find_data_files(
            data_dir,
            splits=[
                "train",
                "test",
                "val",
                "with_no_annotation",
            ],
        )

        # load from local directory
        dataset = ds.load_dataset(dataset_cfg.data_type, data_files=data_files)

        for split in dataset.keys():
            logger.info(f"Load dataset {split}: {len(dataset[split])}")

        vocabulary_file = os.path.join(data_dir, "vocabulary.json")
        with fsspec.open(vocabulary_file, "r") as f:
            vocabulary = json.load(f)

        # Replace categorical fields. (ds.Value -> ds.ClassLabel)
        features = dataset["train"].features.copy()
        for split in vocabulary:
            names = sorted(list(vocabulary[split].keys()))
            if isinstance(features[split], ds.Sequence):
                features[split] = ds.Sequence(ds.ClassLabel(names=names))
            else:
                features[split] = ds.ClassLabel(names=names)
        dataset = dataset.cast(features)
        logger.info(f"Use classes: {names}")
        # CGL: ['embellishment', 'logo', 'text', 'underlay']

    if remove_column_names:
        dataset = dataset.remove_columns(remove_column_names)

    # add transform names as a column since
    # .map -like functions cannot take additional arguments (for transforms)
    assert len(transforms) == len(set(transforms)), "Duplicates in transforms"
    logger.info(f"Use transforms: {transforms}")
    for split in dataset:
        split_len = len(dataset[split])
        new_column = [transforms] * split_len
        dataset[split] = dataset[split].add_column("transforms", new_column)

    features = dataset["train"].features
    assert "id" in features

    # .map scans all the dataset at once (too heavy) thus not recommended
    # we use .with_transform instead
    dataset = dataset.with_transform(_composed_transform)

    return dataset, features
