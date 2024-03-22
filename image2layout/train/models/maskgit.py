import logging
import math
from functools import partial
from typing import Any, Optional

import datasets as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from image2layout.train.helpers.mask import batch_topk_mask, sample_mask
from image2layout.train.helpers.sampling import sample
from omegaconf import DictConfig
from torch import Tensor

from .common.base_model import BaseModel, ConditionalInputsForDiscreteLayout
from .common.common import BaseDecoder, SeqLengthDistribution
from .common.image import ImageEncoder

logger = logging.getLogger(__name__)


# https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py
def _mask_schedule_func(
    ratio: Tensor, schedule: str, total_unknown: Optional[int] = None
) -> Tensor:
    """Generates a mask rate by scheduling mask functions R.
    Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
    training, the input ratio is uniformly sampled; during inference, the input
    ratio is based on the step number divided by the total iteration number: t/T.
    Based on experiements, we find that masking more in training helps.
    Args:
      ratio: The uniformly sampled ratio [0, 1) as input.
      total_unknown: The total number of tokens that can be masked out. For
        example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
        512x512 images.
      method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
        "pow2.5" represents x^2.5
    Returns:
      The mask rate (float).
    """
    assert 0.0 <= torch.min(ratio) and torch.max(ratio) <= 1.0
    exp_dict = {"square": 2, "cubic": 3, "sqrt": 0.5}
    if schedule == "linear":
        mask_ratio: Tensor = 1.0 - ratio
    elif schedule == "cosine":
        mask_ratio = torch.cos(math.pi * 0.5 * ratio)
    elif schedule in exp_dict:
        mask_ratio = 1.0 - torch.pow(ratio, exp_dict[schedule])
    # elif schedule == "log":
    #     mask_ratio = -1.0 * math.log2(ratio) / math.log2(total_unknown)
    # elif schedule == "exp":
    #     mask_ratio = 1.0 - math.exp2(-1.0 * math.log2(total_unknown) * (1 - ratio))
    else:
        raise NotImplementedError

    # time should be slightly bigger than zero
    mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.0)
    return mask_ratio.float()  # type: ignore


class MaskGIT(BaseModel):
    """
    To reproduce
    MaskGIT: Masked Generative Image Transformer (CVPR2022)
    https://arxiv.org/abs/2202.04200
    """

    def __init__(
        self,
        features: ds.Features,
        tokenizer: LayoutSequenceTokenizer,
        d_model: int = 256,
        mask_schedule: str = "linear",
        use_padding_as_vocab: bool = True,
        use_gumbel_noise: bool = True,
        pad_weight: float = 1.0,
        # use_token_critic: bool = False,
    ) -> None:
        super().__init__()
        if use_padding_as_vocab and pad_weight != 1.0:
            assert tokenizer.special_tokens == ["pad", "mask"]
            weight = [1.0 for _ in range(tokenizer.N_total)]
            weight[-2] = pad_weight
            self.loss_fn_ce = nn.CrossEntropyLoss(
                label_smoothing=0.1, weight=torch.tensor(weight)
            )
        else:
            self.loss_fn_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.num_timesteps: int = 50
        self.features = features
        self.tokenizer = tokenizer
        self.use_padding_as_vocab = use_padding_as_vocab
        self.use_gumbel_noise = use_gumbel_noise

        self.mask_schedule_func = partial(_mask_schedule_func, schedule=mask_schedule)
        num_layers = 6
        nhead = 8
        self.encoder = ImageEncoder(
            d_model=d_model,
            nhead=nhead,
            backbone_name="resnet50",
            num_layers=num_layers,
        )
        self.decoder = BaseDecoder(
            d_label=self.tokenizer.N_total,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            pos_emb="layout",
        )
        self.seq_dist = SeqLengthDistribution(tokenizer.max_seq_length)
        self.is_causal = False

        self.init_weight()

    def init_weight(self):
        self.encoder.init_weight()
        self.decoder.init_weight()

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Variable names are following nn.TransformerDecoder to avoid confusion.
        """
        memory = self.encoder(inputs["image"])
        if self.use_padding_as_vocab:
            logits = self.decoder(
                tgt=inputs["seq"],
                memory=memory,
                tgt_key_padding_mask=None,  # attend to all the tokens in tgt
                is_causal=self.is_causal,
            )
        else:
            logits = self.decoder(
                tgt=inputs["seq"],
                memory=memory,
                tgt_key_padding_mask=inputs["tgt_key_padding_mask"],  # ignore [PAD]
                is_causal=self.is_causal,
            )
        return {"logits": logits}

    def train_loss(
        self, inputs: dict, targets: dict, test: bool = False
    ) -> tuple[dict[str, Tensor]]:
        loss_mask = targets["loss_mask"]
        outputs = self(inputs)
        losses = {
            "nll_loss": self.loss_fn_ce(
                outputs["logits"][loss_mask],
                targets["seq"][loss_mask],
            )
        }
        return outputs, losses  # type: ignore

    def sample(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        sampling_cfg: DictConfig = None,
        batch_size: Optional[int] = None,
        return_violation: bool = False,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        use_learned_seq_dist = False  # reasonable in image-conditioned setting

        B = cond.image.size(0)
        S = self.tokenizer.max_token_length
        if (B == 1) and batch_size and batch_size > 1:
            cond = cond.duplicate(n=batch_size)

        sp_tokens = self.tokenizer.special_tokens
        token_mask = self.tokenizer.token_mask
        ids = {key: self.tokenizer.name_to_id(key) for key in sp_tokens}
        N_total = self.tokenizer.N_total
        T = sampling_cfg.get("num_timesteps", 10)

        if cond.seq is not None:
            seq = cond.seq.cpu().clone()
            # **_user will not be updated (kept as reference)
            seq_user = cond.seq.cpu().clone()
            mask_user = cond.mask.cpu().clone()
            if not self.use_padding_as_vocab:
                tgt_key_padding_mask_user = seq == ids["pad"]
        else:
            if use_learned_seq_dist:
                # using seq distribution (not sure if this is correct in image-conditioned case)
                n_elements = self.seq_dist.sample(B) * self.tokenizer.N_var_per_element
                indices = rearrange(torch.arange(S), "s -> 1 s")
                mask = indices < rearrange(n_elements, "b -> b 1")
                seq = torch.full((B, S), fill_value=ids["pad"])  # type: ignore
                seq[mask] = ids["mask"]
                seq_user = seq.clone()  # type: ignore
                mask_user = ~mask.clone()
            else:
                mask = torch.full((B, S), fill_value=True)
                seq = torch.full((B, S), fill_value=ids["pad"])
                mask_user = ~mask.clone()
                seq_user = seq.clone()
                if not self.use_padding_as_vocab:
                    tgt_key_padding_mask_user = ~mask.clone()

        if cond.task in ["c", "cwh", "refinement"]:
            is_element_num_known = True
            element_mask = seq != ids["pad"]
        else:
            is_element_num_known = False

        with torch.no_grad():
            # relatively heavy, make sure to call this only once
            memory = self.encoder(cond.image)

        for t in range(T):
            float_t = torch.full((B,), (t + 1) / T)  # 1/T -> 1.0
            mask_ratio = self.mask_schedule_func(float_t)  # 1.0 -> 0.0
            temperature_at_t = sampling_cfg.get("temperature", 1.0) * (1.0 - float_t)
            is_masked = seq == ids["mask"]

            with torch.no_grad():
                device = memory.device
                inputs = {"tgt": seq.to(device), "memory": memory}
                if not self.use_padding_as_vocab:
                    inputs["tgt_key_padding_mask"] = tgt_key_padding_mask_user.to(
                        device
                    )
                logits = self.decoder(**inputs).cpu()

            invalid = repeat(~token_mask, "s c -> b s c", b=B)
            if is_element_num_known:
                # avoid predicting [PAD]
                pad_mask = repeat(element_mask, "b s -> b s x", x=N_total)
                pad_mask = pad_mask & (
                    rearrange(torch.arange(N_total), "x -> 1 1 x") == ids["pad"]
                )
                invalid = invalid | pad_mask
            logits[invalid] = -float("Inf")

            seq_pred = sample(rearrange(logits, "b s c -> b c s"), sampling_cfg)
            seq_pred = rearrange(seq_pred, "b 1 s -> b s")

            probs = F.softmax(logits, dim=2)
            confidence = torch.gather(
                torch.log(probs),
                2,
                rearrange(seq_pred, "b s -> b s 1"),
            )
            confidence = rearrange(confidence, "b s 1 -> b s")
            if self.use_gumbel_noise:
                # add gumbel noise in choosing tokens
                # https://github.com/google-research/maskgit/blob/cf615d448642942ddebaa7af1d1ed06a05720a91/maskgit/libml/parallel_decode.py#L29
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(confidence) + 1e-30) + 1e-30
                )
                # larger temp. adds more randomness
                confidence += rearrange(temperature_at_t, "b -> b 1") * gumbel_noise

            # non-masked region is kept forever
            # confidence[~is_masked] = CONFIDENCE_OF_KNOWN
            seq = torch.where(is_masked, seq_pred, seq)

            if t < T - 1:
                # re-fill [MASK] for unconfident predictions
                n_elem = reduce(~mask_user, "b s -> b", reduction="sum")
                topk = (n_elem * mask_ratio).long().clamp(min=1)
                is_unconfident, _ = batch_topk_mask(
                    -1.0 * confidence, topk, mask=is_masked
                )
                seq[is_unconfident] = ids["mask"]

            # make sure to use user-defined inputs
            seq[mask_user] = seq_user[mask_user]

        layouts = self.tokenizer.decode(seq)
        if not return_violation:
            return layouts

        return layouts, None

    def preprocess(
        self, inputs: dict[str, Any]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Randomly mask tokens in the input sequence.
        """
        self.seq_dist.update(inputs["mask"])
        data = self.tokenizer.encode(inputs)
        image = torch.cat([inputs["image"], inputs["saliency"]], dim=1)

        B = data["seq"].size(0)
        mask_id = self.tokenizer.name_to_id("mask")
        pad_id = self.tokenizer.name_to_id("pad")
        masked_seq = data["seq"].clone()

        # randomly replace valid tokens with [MASK]s,
        # and get the replaced positions as loss_mask
        mask_ratio = self.mask_schedule_func(ratio=torch.rand((B,)))
        if self.use_padding_as_vocab:
            _mask = torch.full(data["mask"].size(), True)
        else:
            _mask = inputs["mask"]
        loss_mask = sample_mask(_mask, mask_ratio)  # type: ignore
        masked_seq[loss_mask] = mask_id

        new_inputs = {"seq": masked_seq, "image": image}
        targets = {"seq": data["seq"], "loss_mask": loss_mask}
        if not self.use_padding_as_vocab:
            # like typical autoreg. models, ignore [PAD]s in tgt
            new_inputs["tgt_key_padding_mask"] = masked_seq == pad_id

        return new_inputs, targets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt_1 = plt.figure(figsize=(4, 4))
    n_bin = 100
    # x = torch.linspace(0.0, 1.0, 100)
    xs = [i / n_bin for i in range(n_bin + 1)]
    # schedules = ["linear", "cosine", "square", "cubic", "sqrt", "log", "exp"]
    schedules = ["linear", "cosine", "square", "cubic", "sqrt"]
    for schedule in schedules:
        ys = _mask_schedule_func(
            torch.tensor(xs), schedule=schedule, total_unknown=256
        ).tolist()
        plt.plot(xs, ys, label=schedule)
    plt.legend()
    plt.savefig("mask_schedule.pdf")
