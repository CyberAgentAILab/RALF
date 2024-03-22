import logging
from typing import Any

import torch
from torch import LongTensor, Tensor

from .base import BaseMaskAndReplaceDiffusion, DiffusionForwardOutput
from .util import (
    extract,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_onehot_to_index,
)

logger = logging.getLogger(__name__)


class MaskAndReplaceDiffusion(BaseMaskAndReplaceDiffusion):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if self.alpha_init_type == "alpha1":
            N = self.d_label - 1
            at, bt, ct, att, btt, ctt = self.alpha_schedule_partial_func(N=N)
        else:
            print("alpha_init_type is Wrong !! ")

        log_at, log_bt, log_ct = torch.log(at), torch.log(bt), torch.log(ct)
        log_cumprod_at, log_cumprod_bt, log_cumprod_ct = (
            torch.log(att),
            torch.log(btt),
            torch.log(ctt),
        )

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item()
            < 1.0e-5
        )

        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

    def q_pred_one_timestep(self, log_x_t: Tensor, t: Tensor) -> Tensor:  # q(xt|xt_1)
        s = log_x_t.size()
        log_at = extract(self.log_at, t, s)  # type: ignore # at
        log_bt = extract(self.log_bt, t, s)  # type: ignore # bt
        log_ct = extract(self.log_ct, t, s)  # type: ignore # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, s)  # type: ignore  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: Tensor) -> Tensor:  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # type: ignore # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # type: ignore # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # type: ignore # ct~
        log_1_min_cumprod_ct = extract(
            self.log_1_min_cumprod_ct, t, log_x_start.shape
        )  # type: ignore # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),
            ],
            dim=1,
        )

        return log_probs

    def q_posterior(
        self,
        log_x_start: Tensor,
        log_x_t: Tensor,
        t: Tensor,
    ) -> Tensor:  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.d_label - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(
            -1, -1, self.max_token_length
        )

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # type: ignore # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.d_label - 1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # type: ignore # ct
        ct_vector = log_ct.expand(-1, self.d_label - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = (
            self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        )
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_sample(
        self, log_x_start: Tensor, t: LongTensor
    ) -> Tensor:  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def core(
        self, x_start: LongTensor, memory: Tensor, t: LongTensor
    ) -> DiffusionForwardOutput:
        """
        Compute the forward diffusion process while keeping many intermediate states
        """
        log_x_start = index_to_log_onehot(x_start, self.d_label)
        log_x_t = self.q_sample(log_x_start=log_x_start, t=t)
        x_t = log_onehot_to_index(log_x_t)

        # go to p_theta function
        log_x0_recon = self.predict_start(
            log_x_t=log_x_t, memory=memory, t=t
        )  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(
            log_x_start=log_x0_recon, log_x_t=log_x_t, t=t
        )  # go through q(xt_1|xt,x0)

        return DiffusionForwardOutput(
            log_x_start=log_x_start,
            log_x_t=log_x_t,
            log_x0_recon=log_x0_recon,
            log_model_prob=log_model_prob,
            x_t=x_t,
        )
