import re
from typing import Optional, Union

import torch
from einops import repeat
from image2layout.train.global_variables import GEO_KEYS
from image2layout.train.models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)
from .layout_tokenizer import LayoutSequenceTokenizer
from .mask import sample_mask
from .relationships import compute_relation


REFINEMENT_NOISE_STD = 0.01  # std of noise for refinement
EDGE_RATIO = 0.1
COND_TYPES = [
    "c",  # given category (=label), predict position and size (C->P+S)
    "cwh",  # given category (=label) and size, predict position (C+S->P)
    # given a partial layout (i.e., only a few elements),
    #  predict to generate a complete layout (i.e. completion)
    "partial",
    "gt",  # just copy
    # "random",  # random masking
    # given category and noisy position and size, predict accurate position and size
    "refinement",
    "relation",  # given category and some relationships between elements,
    # try to fulfill the relationships as much as possible
    None,
    "none",
    "uncond",
]
VARS = {
    "c": [
        "label",
    ],
    "cwh": ["label", "width", "height"],
    "relation": ["label"],
    "refinement": ["label", "width", "height", "center_x", "center_y"],
    "partial": ["label", "width", "height", "center_x", "center_y"],
}



def get_condition(
    batch: dict,  # sampled from DataLoader
    cond_type: Optional[str] = None,
    tokenizer: Optional[LayoutSequenceTokenizer] = None,
    model_type: str = "Autoreg",
) -> tuple[
    Union[
        ConditionalInputsForDiscreteLayout,
        RetrievalAugmentedConditionalInputsForDiscreteLayout,
        dict,
    ],
    dict,
]:
    """
    For layout-conditional prediction, make an input sequence
    where a [MASK] indicates a token to be predicted
    If cond_type is None, generation is just based on the input image
    Returns:
        cond:
            seq: [B, 5 * max_elem + 1]
            mask: True means valid and special token, False means invalid token.
                invalid token means -1
    """
    assert cond_type in COND_TYPES

    if tokenizer is None:
        # will later processed by BaseGANGenerator.preprocess
        return batch, batch

    if batch["image"].size(1) == 4:
        image = batch["image"]
    else:
        image = torch.cat([batch["image"], batch["saliency"]], dim=1)

    special_keys = tokenizer.special_tokens
    pad_id = tokenizer.name_to_id("pad")
    mask_id = tokenizer.name_to_id("mask") if "mask" in special_keys else -1
    cond = tokenizer.encode(batch)
    B, S = cond["seq"].size()
    C = tokenizer.N_var_per_element

    if cond_type is None or cond_type == "none" or cond_type == "uncond":
        cond = {"seq": None, "mask": None}
    elif cond_type == "partial":
        keep = batch["mask"].clone()
        keep[:, 0] = True  # BOS
        keep[:, 1:] = False  # BOS
        keep = repeat(keep, "b s -> b (s c)", c=C)  # token-level flag

        if "bos" in special_keys:
            # for order-sensitive methods, shift valid condition at the beginning of seq.
            keep = torch.cat([torch.full((B, 1), fill_value=True), keep], dim=-1)
            new_seq = torch.full_like(cond["seq"], mask_id)  # -1
            new_mask = torch.full_like(cond["mask"], False)
            for i in range(B):
                s = cond["seq"][i]
                ind_end = keep[i].sum().item()
                new_seq[i][:ind_end] = s[keep[i]]
                new_mask[i][:ind_end] = True
            cond["seq"] = new_seq  # [bs, 5*max_elem+1]
            cond["mask"] = new_mask  # [bs, 5*max_elem+1]
        else:
            cond["seq"][~keep] = mask_id
            cond["mask"] = keep

    elif cond_type in ["c", "cwh", "relation"]:

        if cond_type == "relation":
            cond = {**cond, **compute_relation(batch, edge_ratio=EDGE_RATIO)}

        keep = torch.full((B, S), False)

        # generate repeating pattern indicating each attribute
        if "bos" in special_keys:
            attr_ind = (torch.arange(S).view(1, S) - 1) % C
            attr_ind[:, 0] = -1  # dummy id for BOS
            keep[:, 0] = True
        else:
            attr_ind = torch.arange(S).view(1, S) % C

        # keep only specified attributes
        for attr_type in VARS[cond_type]:
            ind = tokenizer.var_order.index(attr_type)
            keep |= attr_ind == ind
        cond["seq"][~keep] = mask_id

        # specify number of elements since it is known in the current setting
        cond["seq"][~cond["mask"]] = pad_id
        cond["mask"] = (cond["mask"] & keep) | ~cond["mask"]

    elif cond_type == "gt":
        pass

    elif cond_type == "random":
        ratio = torch.rand((B,))
        loss_mask = sample_mask(torch.full(cond["mask"].size(), True), ratio)

        cond["seq"][loss_mask] = mask_id
        cond["mask"] = ~loss_mask

    elif cond_type == "refinement":
        new_batch = {"label": batch["label"], "mask": batch["mask"]}
        for key in GEO_KEYS:
            noise = torch.normal(0, REFINEMENT_NOISE_STD, size=batch[key].size())
            new_batch[key] = torch.clamp(batch[key] + noise, min=0.0, max=1.0)
            new_batch[key][~batch["mask"]] = 0.0  # just in case

            batch[key] = new_batch[key].clone()

        # Tokenize the perturbed condition
        new_cond = tokenizer.encode(new_batch)

        index = repeat(torch.arange(S), "s -> b s", b=B)
        if "bos" in special_keys:
            index = index - 1
    
        cond = {
            "seq": new_cond["seq"],
            "mask": cond["mask"],  # In refinement, all tokens are valid.
            "seq_observed": new_batch,
        }
    else:
        raise NotImplementedError

    try:
        cond["id"] = torch.tensor(list(map(int, batch["id"])), dtype=torch.long)
    except Exception:
        cond["id"] = batch["id"]

    if "retrieved" in batch.keys():
        class_ = RetrievalAugmentedConditionalInputsForDiscreteLayout
        if isinstance(batch["retrieved"], list):
            assert len(batch["retrieved"]) == 1
            batch["retrieved"] = batch["retrieved"][0]
        cond["retrieved"] = batch["retrieved"]
    else:
        class_ = ConditionalInputsForDiscreteLayout

    return class_(image=image, task=cond_type, **cond), batch
