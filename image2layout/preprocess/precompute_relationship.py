from dataclasses import asdict
from functools import partial
from typing import Any

import cv2
import numpy as np
import seaborn as sns
import torch
from image2layout.train.config import TokenizerConfig
from image2layout.train.data import collate_fn, get_dataset
from image2layout.train.helpers.relationships import (
    RelElement,
    detect_loc_relation_between_element_and_canvas,
    detect_loc_relation_between_elements,
    detect_size_relation,
)
from image2layout.train.helpers.retrieval_dataset_wrapper import RetrievalDatasetWrapper
from image2layout.train.helpers.util import convert_xywh_to_ltrb
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import tqdm

ELEMENTS = {
    "pku": ["logo", "text", "underlay"],
    "cgl": ["embellishment", "logo", "text", "underlay"],
}
PAD_ELEMENT = "pad"


def generate_unique_labels(
    dataset_name: str, labels_tensor: Tensor, masks_tensor: Tensor
) -> list[list[Any]]:
    unique_labels = []
    element_types = ELEMENTS[dataset_name]

    for idx in range(labels_tensor.size(0)):

        labels = labels_tensor[idx].tolist()
        masks = masks_tensor[idx].tolist()
        # Keep track of how many times each label has appeared
        counts = {label: 0 for label in set(labels)}

        tmp_labels = []
        for label, mask in zip(labels, masks):
            if mask is False:
                tmp_labels.append(PAD_ELEMENT)
                continue
            counts[label] += 1
            unique_label = [element_types[label], list(RelElement)[counts[label] - 1]]
            tmp_labels.append(unique_label)

        unique_labels.append(tmp_labels)

    return unique_labels


def describe_relationships(
    data: dict[str, Tensor], dataset_name: str
) -> dict[str, list]:
    output_relationships = {}
    LABEL = generate_unique_labels(dataset_name, data["label"], data["mask"])
    #  [[['text', RelElement.A], ['text', RelElement.B], ...]...,]

    for batch_idx in range(data["label"].size(0)):

        PALETTE = sns.color_palette("deep", 4)
        H = 750
        W = 513
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        results_position_btw_elems = []
        results_position_btw_canvas = []
        results_size = []
        valid_elements = [i for i, m in enumerate(data["mask"][batch_idx]) if m]
        valid_elements = valid_elements[::-1]
        # Element without having "pad"
        data_id = data["id"][batch_idx]

        for idx, i in enumerate(valid_elements):

            label_A = LABEL[batch_idx][i]
            bbox_A = [
                data["center_x"][batch_idx, i].item(),
                data["center_y"][batch_idx, i].item(),
                data["width"][batch_idx, i].item(),
                data["height"][batch_idx, i].item(),
            ]
            lrrb_A = convert_xywh_to_ltrb(bbox_A)
            _label = int(data["label"][batch_idx, i].item())
            C = [int(c * 255.0) for c in PALETTE[_label]]
            cv2.rectangle(
                canvas,
                (int(lrrb_A[0] * W), int(lrrb_A[1] * H)),
                (int(lrrb_A[2] * W), int(lrrb_A[3] * H)),
                C,
                2,
            )

            # Betweeen elements
            for j in valid_elements[idx + 1 :]:

                label_B = LABEL[batch_idx][j]
                bbox_B = [
                    data["center_x"][batch_idx, j].item(),
                    data["center_y"][batch_idx, j].item(),
                    data["width"][batch_idx, j].item(),
                    data["height"][batch_idx, j].item(),
                ]
                position_rel = detect_loc_relation_between_elements(bbox_A, bbox_B)
                size_rel = detect_size_relation(bbox_A, bbox_B)
                results_position_btw_elems.append([*label_A, position_rel, *label_B])
                results_size.append([*label_A, size_rel, *label_B])

            # Betweeen an element and a canvas
            position_rel_canvas = detect_loc_relation_between_element_and_canvas(bbox_A)
            results_position_btw_canvas.append(
                [*label_A, position_rel_canvas, "canvas", "pad"]
            )

        results = (
            results_position_btw_elems + results_size + results_position_btw_canvas
        )
        output_relationships[data_id] = results

    return output_relationships  # type: ignore


def main(dataset_name: str, max_seq_length: int = 10) -> dict[str, Any]:
    tokenizer_cfg = asdict(TokenizerConfig())

    if dataset_name == "pku":
        dataset_dir_name = f"pku{max_seq_length}"
    else:
        dataset_dir_name = dataset_name

    train_cfg = OmegaConf.create(
        {
            "dataset": {
                "max_seq_length": max_seq_length,
                "data_dir": f"/datasets/PosterLayout/{dataset_dir_name}",
                "data_type": "parquet",
                "path": None,
            },
            "data": {
                "transforms": ["image", "sort_label", "sort_lexicographic"],
                "tokenization": False,
            },
            "sampling": {"name": "random", "temperature": 1.0},
            "tokenizer": tokenizer_cfg,
        }
    )
    dataset, _ = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
    )
    dataset_splits: list[str] = list(dataset.keys())

    loaders: dict[str, torch.utils.data.DataLoader] = {}
    collate_fn_partial = partial(collate_fn, max_seq_length=max_seq_length)
    for split in dataset_splits:

        _dataset = RetrievalDatasetWrapper(
            dataset_name=dataset_name,
            dataset=dataset[split],
            db_dataset=dataset["train"],
            split=split,
            top_k=16,
            max_seq_length=max_seq_length,
            retrieval_backbone="saliency",
            random_retrieval=False,
        )
        loaders[split] = torch.utils.data.DataLoader(
            _dataset,
            shuffle=False,
            num_workers=16,
            batch_size=32,
            pin_memory=True,
            collate_fn=collate_fn_partial,
            persistent_workers=False,
            drop_last=False,
        )

    relationships_dic = {}
    for split in ["train", "test", "val"]:
        relationships_dic[split] = {}
        for batch in tqdm(loaders[split], desc=f"Processing {split} split"):
            _output_relationships = describe_relationships(batch, dataset_name)
            relationships_dic[split] = {
                **relationships_dic[split],
                **_output_relationships,
            }
        assert len(relationships_dic[split].keys()) == len(loaders[split].dataset)

    relationships_dic = {
        **relationships_dic["train"],
        **relationships_dic["test"],
        **relationships_dic["val"],
    }

    return relationships_dic


if __name__ == "__main__":
    outputs: dict[str, Any] = {}
    for dataset_name in ["pku", "cgl"]:
        out: dict = main(dataset_name)
        outputs = {**outputs, **out}

    torch.save(
        outputs, "cache/pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt"
    )

# OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 poetry run python image2layout/preprocess/precompute_relationship.py
