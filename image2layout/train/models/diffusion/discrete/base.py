import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from image2layout.train.helpers.sampling import sample
from image2layout.train.models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
)
from image2layout.train.models.diffusion.common import DiscreteDiffusionDecoder
from omegaconf import DictConfig
from torch import LongTensor, Tensor

from .logit_adjustment import update_logits_for_relation
from .util import (
    LOG_EPS,
    alpha_schedule,
    index_to_log_onehot,
    log_categorical,
    log_onehot_to_index,
    mean_except_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class DiffusionForwardOutput:
    log_x_start: Tensor
    log_x_t: Tensor
    log_x0_recon: Tensor
    log_model_prob: Tensor
    x_t: Tensor


class BaseMaskAndReplaceDiffusion(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        nhead: int,
        tokenizer: LayoutSequenceTokenizer,
        num_timesteps: int,
        pos_emb: str,
        auxiliary_loss_weight: float,  # default: 1e-1
        # original setting in code
        # att_1: float = 0.99999,
        # att_T: float = 0.000009,
        # ctt_1: float = 0.000009,
        # ctt_T: float = 0.99999,
        # original setting in paper (assume eps=1e-4)
        att_1: float = 0.999,
        att_T: float = 0.0001,
        ctt_1: float = 0.0001,
        ctt_T: float = 0.9,
    ) -> None:
        super().__init__()
        assert tokenizer.special_tokens == ["pad", "mask"]
        assert tokenizer.id_to_name(tokenizer.N_total - 1) == "mask"

        self.d_label = tokenizer.N_total
        self.max_token_length = tokenizer.max_token_length
        self.num_timesteps = num_timesteps
        self.tokenizer = tokenizer
        self.train_sampling = "gumbel"

        self.alpha_schedule_partial_func = partial(
            alpha_schedule,
            num_timesteps=num_timesteps,
            att_1=att_1,
            att_T=att_T,
            ctt_1=ctt_1,
            ctt_T=ctt_T,
        )

        kwargs = {}
        if pos_emb == "elem_attr":
            kwargs["n_attr_per_elem"] = len(tokenizer.var_order)

        self.model = DiscreteDiffusionDecoder(
            d_label=self.d_label,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            pos_emb=pos_emb,
            **kwargs,
        )

        self.alpha_init_type = "alpha1"
        # self.loss_type = "vb_stochastic"
        # self.parametrization = "x0"
        self.mask_weight = [1.0, 1.0]
        self.adaptive_auxiliary_loss = True

        self.num_timesteps = num_timesteps
        self.auxiliary_loss_weight = auxiliary_loss_weight

        self.diffusion_acc_list = [0.0] * self.num_timesteps
        self.diffusion_keep_list = [0.0] * self.num_timesteps
        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))
        self.zero_vector = None

    def init_weight(self) -> None:
        self.model.init_weight()

    def multinomial_kl(
        self, log_prob1: Tensor, log_prob2: Tensor
    ) -> Tensor:  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, *args: Any, **kwargs: Any):  # type: ignore
        """
        q(xt|xt_1)
        """
        raise NotImplementedError

    def q_pred(self, *args: Any, **kwargs: Any):  # type: ignore
        """
        q(xt|x0)
        """
        raise NotImplementedError

    def predict_start(
        self, log_x_t: Tensor, memory: Tensor, t: Tensor
    ) -> Tensor:  # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        out = self.model(tgt=x_t, memory=memory, timestep=t)

        out = out[:, :, :-1]  # ignore MASK
        out = rearrange(out, "b s c -> b c s")

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.d_label - 1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = (
                torch.zeros(batch_size, 1, self.max_token_length).type_as(log_x_t) - 70
            )
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(
        self, log_x_start: Tensor, log_x_t: Tensor, t: Tensor
    ) -> Tensor:  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        raise NotImplementedError

    @torch.no_grad()
    def p_sample(
        self,
        log_x: Tensor,
        t: LongTensor,
        sampling_cfg: Optional[DictConfig] = None,
    ) -> None:
        # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob = self.p_pred(log_x, t)

        # for compatibility with other approaches
        out_index = sample(model_log_prob, sampling_cfg)
        out_index = rearrange(out_index, "b 1 s -> b s")
        out = index_to_log_onehot(out_index, self.d_label)

        return out

    def log_sample_categorical(
        self, logits: Tensor
    ) -> Tensor:  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.d_label)
        return log_sample

    def q_sample(self, *args: Any, **kwargs: Any):  # type: ignore
        """
        diffusion step, q(xt|x0) and sample xt
        """
        raise NotImplementedError

    def sample_time(
        self, b: int, device: torch.device, method: str = "uniform"
    ) -> tuple[Tensor, Tensor]:
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def sample_single_step(
        self,
        log_z: Tensor,
        memory: Tensor,
        model_t: Tensor,
        skip_step: int,
        cond: ConditionalInputsForDiscreteLayout,
        sampling_cfg: Optional[DictConfig] = None,
    ):
        # note: some part in this func. may require grads
        # (e.g., task:relation requires grads w.r.t. predicted logits for logit adjustment)

        device = log_z.device
        with torch.no_grad():
            # Infer x0 at first
            log_x_recon = self.predict_start(log_x_t=log_z, memory=memory, t=model_t)

            # add less noise!
            time_difference = getattr(sampling_cfg, "time_difference", 0.0)
            if time_difference > 0.0:
                T = self.num_timesteps
                noise_t = torch.clamp(
                    model_t - int(T * time_difference), 0, T - 1
                ).long()
            else:
                noise_t = model_t.clone()

            if skip_step > 0:
                if noise_t[0].item() > skip_step:
                    model_log_prob = self.q_posterior(
                        log_x_start=log_x_recon, log_x_t=log_z, t=noise_t - skip_step
                    )
                else:
                    model_log_prob = self.q_posterior(
                        log_x_start=log_x_recon, log_x_t=log_z, t=noise_t
                    )
            else:
                # model_log_prob = self.p_pred(log_z, t)
                model_log_prob = self.q_posterior(
                    log_x_start=log_x_recon, log_x_t=log_z, t=noise_t
                )

        # adjust logits distribution based on some gradient in logit space
        if cond.seq is not None:
            # impose strong user-specified constraints by replacement
            if cond.mask is not None:
                with torch.no_grad():
                    strong_mask = rearrange(cond.mask, "b s -> b 1 s")
                    strong_log_prob = index_to_log_onehot(cond.seq, self.d_label)
                    model_log_prob = torch.where(
                        strong_mask, strong_log_prob, model_log_prob
                    )

            # logit adjustment by hand-crafted rules
            if cond.task == "refinement":
                with torch.no_grad():
                    model_log_prob[cond.weak_mask] += cond.weak_logits[cond.weak_mask]  # type: ignore

            # logit adjustment by gradients from loss functions
            if cond.task == "relation":
                t = model_t[0].item()
                model_log_prob = update_logits_for_relation(
                    t=t,
                    cond=cond,
                    model_log_prob=model_log_prob,
                    tokenizer=self.tokenizer,
                    sampling_cfg=sampling_cfg,
                )

            # disable [PAD] when the number of elements is known
            if cond.task in ["c", "cwh", "refinement", "relation"]:
                with torch.no_grad():
                    step = self.tokenizer.N_var_per_element
                    B, S = cond.seq.size()
                    pad_id = self.tokenizer.name_to_id("pad")
                    attr_indexes = repeat(torch.arange(S), "s -> b s", b=B).to(device)
                    pad_mask = (attr_indexes % step != 0) & (cond.seq != pad_id)
                    pad_mask = repeat(pad_mask, "b s -> b c s", c=self.d_label)
                    index = rearrange(torch.arange(self.d_label), "c -> 1 c 1")
                    pad_mask = pad_mask & (index.to(device) == pad_id)
                    model_log_prob[pad_mask] = LOG_EPS

        with torch.no_grad():
            out_index = sample(model_log_prob, sampling_cfg)
            out_index = rearrange(out_index, "b 1 s -> b s")
            log_z = index_to_log_onehot(out_index, self.d_label)

        return log_z

    def core(self, *args: Any, **kwargs: Any):  # type: ignore
        """
        Compute the forward diffusion process while keeping many intermediate states
        """
        raise NotImplementedError

    def forward(
        self, tgt: torch.LongTensor, memory: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        b = tgt.size(0)
        device = tgt.device

        x_start = tgt
        t, pt = self.sample_time(b, device, "importance")
        out: DiffusionForwardOutput = self.core(x_start=x_start, memory=memory, t=t)

        # compute acc list
        self.update_list(out=out, x_start=x_start, t=t)

        # compute log_true_prob now
        log_true_prob = self.q_posterior(
            log_x_start=out.log_x_start, log_x_t=out.log_x_t, t=t
        )
        kl = self.multinomial_kl(log_true_prob, out.log_model_prob)
        mask_region = (out.x_t == self.d_label - 1).float()
        mask_weight = (
            mask_region * self.mask_weight[0]
            + (1.0 - mask_region) * self.mask_weight[1]
        )
        kl = kl * mask_weight
        kl = mean_except_batch(kl)

        decoder_nll = -log_categorical(out.log_x_start, out.log_model_prob)
        decoder_nll = mean_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1.0 - mask) * kl

        self.update_Lt(kl_loss=kl_loss, t=t)
        loss1 = kl_loss / pt  # Upweigh loss term of the kl
        losses = {"kl_loss": loss1.mean()}

        if self.auxiliary_loss_weight > 0.0:
            kl_aux = self.multinomial_kl(
                out.log_x_start[:, :-1, :], out.log_x0_recon[:, :-1, :]
            )
            kl_aux = kl_aux * mask_weight
            kl_aux = mean_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1.0 - mask) * kl_aux
            if self.adaptive_auxiliary_loss:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            losses["aux_loss"] = loss2.mean()

        outputs = {"logits": out.log_model_prob}
        return outputs, losses

    def update_Lt(self, kl_loss: Tensor, t: Tensor) -> None:
        """
        Collect stats for importance sampling
        """
        if self.training:
            Lt2 = kl_loss.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

    def update_list(
        self,
        out: DiffusionForwardOutput,
        x_start: torch.LongTensor,
        t: torch.LongTensor,
    ) -> None:
        x0_recon = log_onehot_to_index(out.log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(out.log_model_prob)
        xt_recon = log_onehot_to_index(out.log_x_t)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (
                x0_recon[index] == x0_real[index]
            ).sum().cpu() / x0_real.size()[1]
            self.diffusion_acc_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            )
            same_rate = (
                xt_1_recon[index] == xt_recon[index]
            ).sum().cpu() / xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9
            )
