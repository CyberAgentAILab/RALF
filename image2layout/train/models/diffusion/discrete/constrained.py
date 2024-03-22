import logging
from typing import Any

import torch
from einops import rearrange
from image2layout.train.global_variables import GEO_KEYS
from torch import Tensor

from .base import BaseMaskAndReplaceDiffusion, DiffusionForwardOutput
from .pf_converter import Converter
from .util import (
    extract,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_onehot_to_index,
)

logger = logging.getLogger(__name__)


class ConstrainedMaskAndReplaceDiffusion(BaseMaskAndReplaceDiffusion):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """
        See BaseMaskAndReplaceDiffusion for keyword arguments.
        """
        super().__init__(**kwargs)
        self.converter = Converter(self.tokenizer)

        # set vocabulari size for each corruption matrix (w/ pad)
        self.mat_size = {"label": self.tokenizer.N_label + 2}
        num_bin = self.tokenizer.N_bbox_per_var
        for key in GEO_KEYS:
            self.mat_size.update({key: num_bin + 2})

        for key in self.tokenizer.var_order:
            if self.alpha_init_type == "alpha1":
                N = self.mat_size[key] - 1
                at, bt, ct, att, btt, ctt = self.alpha_schedule_partial_func(N=N)
            else:
                print("alpha_init_type is Wrong !! ")
                raise NotImplementedError

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
            self.register_buffer(f"{key}_log_at", log_at.float())
            self.register_buffer(f"{key}_log_bt", log_bt.float())
            self.register_buffer(f"{key}_log_ct", log_ct.float())
            self.register_buffer(f"{key}_log_cumprod_at", log_cumprod_at.float())
            self.register_buffer(f"{key}_log_cumprod_bt", log_cumprod_bt.float())
            self.register_buffer(f"{key}_log_cumprod_ct", log_cumprod_ct.float())
            self.register_buffer(f"{key}_log_1_min_ct", log_1_min_ct.float())
            self.register_buffer(
                f"{key}_log_1_min_cumprod_ct", log_1_min_cumprod_ct.float()
            )

    def q_pred_one_timestep(
        self, log_x_t: Tensor, t: Tensor, key: str
    ) -> Tensor:  # q(xt|xt_1)
        s = log_x_t.size()
        log_at = extract(getattr(self, f"{key}_log_at"), t, s)  # at
        log_bt = extract(getattr(self, f"{key}_log_bt"), t, s)  # bt
        log_ct = extract(getattr(self, f"{key}_log_ct"), t, s)  # ct
        log_1_min_ct = extract(getattr(self, f"{key}_log_1_min_ct"), t, s)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: Tensor, key: str) -> Tensor:  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        s = log_x_start.size()
        log_cumprod_at = extract(getattr(self, f"{key}_log_cumprod_at"), t, s)  # at~
        log_cumprod_bt = extract(getattr(self, f"{key}_log_cumprod_bt"), t, s)  # bt~
        log_cumprod_ct = extract(getattr(self, f"{key}_log_cumprod_ct"), t, s)  # ct~
        log_1_min_cumprod_ct = extract(
            getattr(self, f"{key}_log_1_min_cumprod_ct"), t, s
        )  # 1-ct~

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
        device = log_x_start.device
        if self.converter.get_device() == torch.device("cpu"):
            self.converter.to(device)

        log_x_start_full, log_x_t_full = log_x_start, log_x_t  # for API compatibiliry

        batch_size = log_x_start_full.size()[0]
        step = self.tokenizer.N_var_per_element

        index_x_t_full = log_onehot_to_index(log_x_t_full)
        log_one_vector_full = torch.zeros(batch_size, 1, 1).type_as(log_x_t_full)
        seq_len = self.max_token_length // step
        log_zero_vector_full = torch.log(log_one_vector_full + 1.0e-30).expand(
            -1, -1, seq_len
        )
        mask_reshaped = rearrange(
            index_x_t_full == self.tokenizer.name_to_id("mask"),
            "b (s x) -> b s x",
            s=seq_len,
            x=step,
        )

        log_EV_xtmin_given_xt_given_xstart_full = []
        for i, key in enumerate(self.tokenizer.var_order):
            mask = mask_reshaped[..., i].unsqueeze(1)
            log_x_start = self.converter.f_to_p_log(log_x_start_full[..., i::step], key)
            log_x_t = self.converter.f_to_p_log(log_x_t_full[..., i::step], key)
            log_qt = self.q_pred(log_x_t, t, key)  # q(xt|x0)

            log_qt = log_qt[:, :-1, :]
            log_cumprod_ct = extract(
                getattr(self, f"{key}_log_cumprod_ct"), t, log_x_t.size()
            )  # ct~
            ct_cumprod_vector = log_cumprod_ct.expand(-1, self.mat_size[key] - 1, -1)
            log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

            log_qt_one_timestep = self.q_pred_one_timestep(
                log_x_t, t, key
            )  # q(xt|xt_1)

            log_qt_one_timestep = torch.cat(
                (log_qt_one_timestep[:, :-1, :], log_zero_vector_full), dim=1
            )
            log_ct = extract(getattr(self, f"{key}_log_ct"), t, log_x_t.size())  # ct
            ct_vector = log_ct.expand(-1, self.mat_size[key] - 1, -1)
            ct_vector = torch.cat((ct_vector, log_one_vector_full), dim=1)
            log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

            # below just does log(ab/c)=loga+logb-logc in eq.5 of VQDiffusion
            q = log_x_start[:, :-1, :] - log_qt
            q = torch.cat((q, log_zero_vector_full), dim=1)
            q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
            q = q - q_log_sum_exp
            log_EV_xtmin_given_xt_given_xstart = (
                self.q_pred(q, t - 1, key) + log_qt_one_timestep + q_log_sum_exp
            )
            log_EV_xtmin_given_xt_given_xstart = torch.clamp(
                log_EV_xtmin_given_xt_given_xstart, -70, 0
            )
            log_EV_xtmin_given_xt_given_xstart_full.append(
                self.converter.p_to_f_log(log_EV_xtmin_given_xt_given_xstart, key)
            )

        log_EV_xtmin_given_xt_given_xstart_final: Tensor = torch.stack(
            log_EV_xtmin_given_xt_given_xstart_full, dim=-1
        ).view(
            batch_size, self.d_label, -1
        )  # type: ignore

        return log_EV_xtmin_given_xt_given_xstart_final

    def log_sample_categorical(  # type: ignore
        self, logits: Tensor, key: str
    ) -> Tensor:
        """
        use gumbel to sample onehot vector from log probability
        """
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sampled = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sampled, self.mat_size[key])
        return log_sample

    def q_sample(
        self, log_x_start: Tensor, t: Tensor, key: str
    ) -> Tensor:  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, key)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0, key)

        return log_sample

    def core(
        self, x_start: Tensor, memory: Tensor, t: Tensor
    ) -> DiffusionForwardOutput:
        """
        Compute the forward diffusion process while keeping many intermediate states
        """
        if self.converter.get_device() == torch.device("cpu"):
            self.converter.to(x_start.device)

        B, S = x_start.size()[:2]
        X = self.tokenizer.N_var_per_element
        log_x_start = index_to_log_onehot(x_start, self.d_label)
        x_start_reshaped = self.converter.f_to_p_id_all(
            rearrange(x_start, "b (s x) -> b s x", s=S // X, x=X)
        )

        log_x_t_list, xt_list = [], []
        for i, key in enumerate(self.tokenizer.var_order):
            log_x_start_partial = index_to_log_onehot(
                x_start_reshaped[..., i], self.mat_size[key]
            )
            log_x_t_partial = self.q_sample(
                log_x_start=log_x_start_partial, t=t, key=key
            )
            log_x_t_list.append(self.converter.p_to_f_log(log_x_t_partial, key))
            xt_list.append(log_onehot_to_index(log_x_t_partial))

        x_t = self.converter.p_to_f_id_all(torch.stack(xt_list, dim=-1)).view(B, -1)  # type: ignore
        log_x_t = torch.stack(log_x_t_list, dim=-1).view(B, self.d_label, -1)

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
