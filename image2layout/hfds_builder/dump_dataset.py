import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets as ds
import numpy as np
from helpers.cgl import read_cgl
from helpers.global_variables import (
    EMPTY_DATA,
    HEIGHT_RESIZE_IMAGE,
    HFDS_FEATURES,
    WIDTH_RESIZE_IMAGE,
)
from helpers.pku import read_pku
from helpers.util import Element, Sample
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="pku",
        choices=["pku", "cgl"],
        help="Kind of pipeline to invoke.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        type=str,
        help="Input file path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--num-shards",
        default=8,
        type=int,
        help="Number of shards.",
    )
    parser.add_argument(
        "--log_level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "pku":
        reader = read_pku
    elif args.dataset_type == "cgl":
        reader = read_cgl
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    identifier_split_mapping = {}
    for path in Path(f"data_splits/splits/{args.dataset_type}").glob("*.txt"):
        name = path.stem
        with path.open("r") as f:
            lines = [t.strip() for t in f.readlines()]

            prefix = "test" if name == "with_no_annotation" else "train"
            for line in lines:
                identifier = f"{prefix}/{line}.png"
                identifier_split_mapping[identifier] = name

    kwargs = {
        "dataset_root": args.dataset_root,
        "image_size": (WIDTH_RESIZE_IMAGE, HEIGHT_RESIZE_IMAGE),
    }

    data = defaultdict(list)
    for sample in tqdm(
        reader(args.dataset_root), desc="Transforming dataset (e.g., resizing images)"
    ):
        record = _make_record(sample, **kwargs)
        id_ = record["identifier"]
        if id_ in identifier_split_mapping:
            new_split = identifier_split_mapping[id_]
        elif id_.endswith(".jpg"):
            id_ = id_.replace(".jpg", ".png")
            new_split = identifier_split_mapping[id_]
        else:
            raise NotImplementedError

        for key in ["split", "identifier"]:
            del record[key]
        data[new_split].append(record)

    logger.info("Writing to parquet files ...")
    for split, records in data.items():
        logger.info(f"{split=}, {len(records)=}")
        dataset = ds.Dataset.from_list(records, features=HFDS_FEATURES)
        num_shards = args.num_shards if split == "train" else 1
        for index in range(num_shards):
            shard = dataset if num_shards == 1 else dataset.shard(num_shards, index)
            parquet_name = f"{split}-{index:05d}-of-{num_shards:05d}.parquet"
            shard.to_parquet(str(output_dir / parquet_name))
    logger.info("Done.")


def _make_record(
    sample: Sample,
    dataset_root: str,
    image_size: tuple[int, int],
) -> dict:
    output = {}
    if len(sample.elements) == 0:
        output.update(EMPTY_DATA)
    else:
        output.update(_unpack_elements(sample.elements))

    paths = {}
    for key in ["input", "saliency", "saliency_sub"]:
        paths[key] = os.path.join(
            dataset_root, "image", sample.split, key, f"{sample.id}.png"
        )

    # early resize of image-like data to improve throughput
    image = Image.open(paths["input"]).convert("RGB")
    output["image"] = image.resize(image_size)

    saliency = Image.open(paths["saliency"]).convert("L")
    saliency_sub = Image.open(paths["saliency_sub"]).convert("L")
    saliency = Image.fromarray(np.maximum(np.array(saliency), np.array(saliency_sub)))
    output["saliency"] = saliency.resize(image_size)

    # remove some keys
    for key in sample._fields:
        if key not in [
            "elements",
        ]:
            output[key] = getattr(sample, key)

    return output


def _unpack_elements(elements: list[Element]) -> dict[str, list[Any]]:
    output = defaultdict(list)
    for element in elements:
        for key in ["center_x", "center_y", "width", "height"]:
            output[key].append(getattr(element.coordinates, key))
        output["label"].append(element.label)
    return output


if __name__ == "__main__":
    main()
