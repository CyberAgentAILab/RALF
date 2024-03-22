import torch
from torch import Tensor

BBOX_KEYS = ["center_x", "center_y", "width", "height"]


def generate_fake_and_real(
    batch: dict[str, Tensor], std: float = 0.05
) -> dict[str, Tensor]:
    B, S = batch["label"].size()
    is_fake = torch.randint(0, 2, (B,)).bool()

    for key in BBOX_KEYS:
        noise = torch.normal(0, std, size=(B, S))
        value = batch[key] + noise
        batch[key][is_fake] = value[is_fake]
        batch[key][~batch["mask"]] = 0.0  # fill by padded values

    batch["is_real"] = (~is_fake).float()
    return batch
