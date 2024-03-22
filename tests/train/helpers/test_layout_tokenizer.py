import os
import random
import string
from typing import Any

import datasets as ds
import torch
from einops import rearrange, reduce
from image2layout.train.global_variables import GEO_KEYS, PRECOMPUTED_WEIGHT_DIR
from image2layout.train.helpers.layout_tokenizer import CHOICES, LayoutSequenceTokenizer
from tests.util import repeat_func


def _setup_dummy_cfg() -> dict[str, Any]:
    """
    Config for LayoutSequenceTokenizer
    """
    special_tokens = ["pad"]
    if random.random() > 0.5:
        special_tokens.extend(["bos", "eos"])
    if random.random() > 0.5:
        special_tokens.extend(["mask"])
    pad_until_max = True if random.random() > 0.5 and "pad" in special_tokens else False

    data = {
        "num_bin": 2 ** random.randint(4, 8),
        "special_tokens": special_tokens,
        "pad_until_max": pad_until_max,
    }
    for (key, seq) in CHOICES.items():
        data[key] = random.choice(seq)  # type: ignore
    return data


def _setup_dummmy_inputs(
    batch_size: int, max_seq_length: int, num_labels: int
) -> dict[str, torch.Tensor]:
    seq_len = torch.randint(1, max_seq_length, (batch_size, 1))
    N = int(seq_len.max().item())
    inputs = {
        "label": torch.randint(num_labels, (batch_size, N)),
        "center_x": torch.rand((batch_size, N)),
        "center_y": torch.rand((batch_size, N)),
        "width": torch.rand((batch_size, N)),
        "height": torch.rand((batch_size, N)),
        "mask": seq_len > torch.arange(0, N).view(1, N),
    }
    inputs["label"][~inputs["mask"]] = 0
    for key in GEO_KEYS:
        inputs[key][~inputs["mask"]] = 0.0
    return inputs


@repeat_func(100)
def test_layout_tokenizer() -> None:
    batch_size = random.randint(1, 10)
    max_seq_length = random.randint(2, 32)
    num_labels = random.randint(1, 10)
    names = list(string.ascii_lowercase)[:num_labels]

    label_feature = ds.ClassLabel(num_classes=num_labels, names=names)
    kwargs = _setup_dummy_cfg()

    if kwargs["geo_quantization"] == "kmeans":
        weight_path = os.path.join(
            PRECOMPUTED_WEIGHT_DIR, "clustering", "pku10_kmeans_train_clusters.pkl"
        )
    else:
        weight_path = None

    tokenizer = LayoutSequenceTokenizer(
        label_feature=label_feature,
        max_seq_length=max_seq_length,
        weight_path=weight_path,
        **kwargs,
    )

    inputs = _setup_dummmy_inputs(
        batch_size=batch_size, max_seq_length=max_seq_length, num_labels=num_labels
    )
    special_tokens = tokenizer.special_tokens
    seq_len = reduce(inputs["mask"], "b s -> b 1", reduction="sum")

    seq = tokenizer.encode(inputs)["seq"]
    if "bos" in special_tokens and "eos" in special_tokens:
        seq = seq[:, 1:]  # remove BOS

    new_inputs = tokenizer.decode(seq)
    if tokenizer.pad_until_max:
        indexes = torch.arange(0, max_seq_length)
        new_inputs["mask"] = seq_len > rearrange(indexes, "s -> 1 s")

    new_seq = tokenizer.encode(new_inputs)["seq"]
    if "bos" in special_tokens and "eos" in special_tokens:
        new_seq = new_seq[:, 1:]  # remove BOS

    for k in inputs:
        if tokenizer.pad_until_max:
            s = inputs[k].size(1)
        else:
            s = new_inputs[k].size(1)

        if inputs[k].dtype == torch.float32:
            # due to bbox quantization error, very roughly check results
            atol = 1.0 / tokenizer.N_bbox_per_var
            if tokenizer.geo_quantization == "linear":
                assert torch.all(
                    torch.isclose(inputs[k][:, :s], new_inputs[k][:, :s], atol=atol)
                ).item()
            elif tokenizer.geo_quantization == "kmeans":
                # in kmeans case, it's very hard to define atol
                pass
        else:
            assert torch.all(inputs[k][:, :s] == new_inputs[k][:, :s]).item()

    assert torch.all(seq == new_seq).item()


if __name__ == "__main__":
    test_layout_tokenizer()
