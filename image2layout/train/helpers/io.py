import copy
import logging
import os
from typing import Optional

import fsspec
import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_dim_model(backbone_cfg: DictConfig) -> int:
    """
    It takes hierarchical config for
    - trainer.models.transformer_utils import TransformerEncoder
    and get a number of dimension inside the Transformer
    """
    result = None
    for key, value in backbone_cfg.items():
        if key == "d_model":
            result = value
        elif isinstance(value, DictConfig):
            x = get_dim_model(value)
            if x:
                result = x
    return result


def shrink(backbone_cfg: DictConfig, mult: float) -> DictConfig:
    """
    Rescale dimension of a model linearly
    """
    new_backbone_cfg = copy.deepcopy(backbone_cfg)
    for key in ["d_model", "dim_feedforward"]:
        dim = int(mult * new_backbone_cfg.encoder_layer[key])
        new_backbone_cfg.encoder_layer[key] = dim
    return new_backbone_cfg


def load_model(
    model: nn.Module,
    ckpt_dir: str,
    device: torch.device,
    best_or_final: str = "best",
    prefix: Optional[str] = None,
) -> nn.Module:
    if prefix:
        model_path = os.path.join(ckpt_dir, f"{prefix}_{best_or_final}_model.pt")
    else:
        model_path = os.path.join(ckpt_dir, f"{best_or_final}_model.pt")
    with fsspec.open(str(model_path), "rb") as file_obj:
        model.load_state_dict(torch.load(file_obj, map_location=device))
    return model


def save_model(
    model: nn.Module,
    ckpt_dir: str,
    best_or_final: str = "best",
    prefix: Optional[str] = None,
) -> None:
    if prefix:
        model_path = os.path.join(ckpt_dir, f"{prefix}_{best_or_final}_model.pt")
    else:
        model_path = os.path.join(ckpt_dir, f"{best_or_final}_model.pt")
    logger.info(f"Save weight {model_path=}")
    with fsspec.open(str(model_path), "wb") as file_obj:
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), file_obj)
        else:
            torch.save(model.state_dict(), file_obj)
    return model
