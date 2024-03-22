import math

import datasets as ds
import torch
from image2layout.train.global_variables import GEO_KEYS
from image2layout.train.helpers.metric import (
    compute_overlay,
    compute_underlay_effectiveness,
)

REL_TOL = 1e-4
_feature_label = ds.ClassLabel(names=["text", "logo", "underlay"])


def _case_to_batch(case: dict) -> dict[str, torch.Tensor]:
    batch = {}
    batch["label"] = torch.tensor(_feature_label.str2int(case["example"]["label"]))
    for key in GEO_KEYS:
        batch[key] = torch.tensor(case["example"][key])
    batch["mask"] = torch.ones_like(batch["label"], dtype=torch.bool)
    batch = {k: v.unsqueeze(0) for (k, v) in batch.items()}  # make batch
    return batch


def test_compute_underlay_effectiveness() -> None:
    cases = [
        # a2 is completely inside a1
        {
            "example": {
                "label": ["text", "underlay"],
                "center_x": [0.5, 0.5],
                "center_y": [0.5, 0.5],
                "width": [0.2, 0.4],
                "height": [0.2, 0.4],
            },
            "expected": {
                "underlay_effectiveness_loose": 1.0,
                "underlay_effectiveness_strict": 1.0,
            },
        },
        # a2 is completely outside a1
        {
            "example": {
                "label": ["text", "underlay"],
                "center_x": [0.1, 0.9],
                "center_y": [0.1, 0.9],
                "width": [0.2, 0.2],
                "height": [0.2, 0.2],
            },
            "expected": {
                "underlay_effectiveness_loose": 0.0,
                "underlay_effectiveness_strict": 0.0,
            },
        },
        # a2 is overlapping with a1
        {
            "example": {
                "label": ["text", "underlay"],
                "center_x": [0.5, 0.5],
                "center_y": [0.5, 0.5],
                "width": [0.2, 0.6],
                "height": [0.6, 0.2],
            },
            "expected": {
                "underlay_effectiveness_loose": 1 / 3,
                "underlay_effectiveness_strict": 0.0,
            },
        },
        # a2 is overlapping with a1, a3 is completely inside a2
        {
            "example": {
                "label": ["text", "underlay", "text"],
                "center_x": [0.5, 0.5, 0.5],
                "center_y": [0.5, 0.5, 0.5],
                "width": [0.2, 0.6, 0.3],
                "height": [0.6, 0.2, 0.1],
            },
            "expected": {
                "underlay_effectiveness_loose": 1.0,
                "underlay_effectiveness_strict": 1.0,
            },
        },
        # a1 < a2 < a3
        # (waste of underlay, but looks ok for this metric)
        {
            "example": {
                "label": ["text", "underlay", "underlay"],
                "center_x": [0.5, 0.5, 0.5],
                "center_y": [0.5, 0.5, 0.5],
                "width": [0.2, 0.3, 0.4],
                "height": [0.2, 0.3, 0.4],
            },
            "expected": {
                "underlay_effectiveness_loose": 1.0,
                "underlay_effectiveness_strict": 1.0,
            },
        },
    ]
    for case in cases:
        batch = _case_to_batch(case)
        predicted = compute_underlay_effectiveness(batch, feature_label=_feature_label)
        for key, value in case["expected"].items():
            assert math.isclose(value, predicted[key][0], rel_tol=REL_TOL), (
                value,
                predicted[key][0],
            )


def test_compute_overlay_effectiveness() -> None:
    cases = [
        {
            "example": {
                "label": ["text", "text", "text"],
                "center_x": [0.3, 0.5, 0.7],
                "center_y": [0.5, 0.5, 0.5],
                "width": [0.4, 0.4, 0.4],
                "height": [0.4, 0.4, 0.4],
            },
            "expected": {
                "overlay": (1 / 3 + 1 / 3 + 0) / 3,
            },
        },
        {
            "example": {
                "label": ["text"],
                "center_x": [0.3],
                "center_y": [0.5],
                "width": [0.4],
                "height": [0.4],
            },
            "expected": {
                "overlay": [],
            },
        },
        {
            "example": {
                "label": ["underlay"],
                "center_x": [0.3],
                "center_y": [0.5],
                "width": [0.4],
                "height": [0.4],
            },
            "expected": {
                "overlay": [],
            },
        },
    ]
    for case in cases:
        batch = _case_to_batch(case)
        predicted = compute_overlay(batch, feature_label=_feature_label)
        for key, value in case["expected"].items():
            pred = predicted[key]
            if len(pred) > 0:
                assert math.isclose(value, pred[0], rel_tol=REL_TOL), (value, pred[0])
            else:
                # expecting no result
                assert len(value) == 0


if __name__ == "__main__":
    test_compute_underlay_effectiveness()
    test_compute_overlay_effectiveness()

# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 poetry run python -m tests.train.helpers.test_metric
