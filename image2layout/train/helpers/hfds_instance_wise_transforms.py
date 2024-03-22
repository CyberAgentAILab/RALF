"""
This is a collection of transform function for an example obtained from Huggingface Dataset.
The example is a dict and expected to have the following keys.
(required)
- label
- center_x
- center_y
- width
- height
(optional)
- id
- image
- saliency

"""

import random
from typing import Any

from torch import Tensor
from torchvision.transforms.functional import to_tensor

from .util import argsort


def reorganize(inputs: dict, indexes: list[int]) -> dict:
    assert isinstance(inputs, dict)
    assert isinstance(indexes, list)

    for key, value in inputs.items():
        if key in ["transforms", "retrieved", "id"]:
            continue

        if isinstance(value, list):
            inputs[key] = [value[ind] for ind in indexes]

    return inputs


def image_transform(inputs: dict) -> dict:
    assert isinstance(inputs, dict)

    if "image" in inputs:
        inputs["image"] = to_tensor(inputs["image"])
    if "saliency" in inputs:
        inputs["saliency"] = to_tensor(inputs["saliency"])
    return inputs


def shuffle_transform(inputs: dict) -> dict:
    assert isinstance(inputs, dict)
    if (N := len(inputs["label"])) == 0:
        return inputs
    else:
        indexes = list(range(N))
        indexes_shuffled = random.sample(indexes, N)
        return reorganize(inputs, indexes_shuffled)


def sort_label_transform(inputs: dict) -> dict:
    assert isinstance(inputs, dict)
    if len(inputs["label"]) == 0:
        return inputs
    else:
        indexes = argsort(inputs["label"])
        return reorganize(inputs, indexes)


def get_indexes_for_lexicographic_sort(inputs: dict) -> list[int]:
    assert isinstance(inputs, dict)
    left = [a - b / 2.0 for (a, b) in zip(inputs["center_x"], inputs["width"])]
    top = [a - b / 2.0 for (a, b) in zip(inputs["center_y"], inputs["height"])]

    left = [x.tolist() if isinstance(x, Tensor) else x for x in left]
    top = [x.tolist() if isinstance(x, Tensor) else x for x in top]

    _zip = zip(*sorted(enumerate(zip(top, left)), key=lambda c: c[1:]))
    indexes = list(list(_zip)[0])

    return indexes


def sort_lexicographic_transform(inputs: dict) -> dict:
    if len(inputs["center_x"]) == 0:
        return inputs
    else:
        indexes = get_indexes_for_lexicographic_sort(inputs)
        return reorganize(inputs, indexes)


HFDS_INSTANCE_WISE_TRANSFORM_FACTORY = {
    "image": image_transform,
    "sort_label": sort_label_transform,
    "sort_lexicographic": sort_lexicographic_transform,
    "shuffle": shuffle_transform,
}


def hfds_instance_wise_trasnform_factory(transform: str) -> Any:
    return HFDS_INSTANCE_WISE_TRANSFORM_FACTORY[transform]
