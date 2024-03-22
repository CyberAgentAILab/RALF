import functools
import logging
import random
from abc import abstractmethod
from typing import Optional

import datasets as ds
import torch
import torch.nn as nn
from einops import rearrange, repeat
from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from image2layout.train.helpers.sampling import sample
from omegaconf import DictConfig
from torch import Tensor

from ..helpers.task import COND_TYPES, get_condition
from .common.base_model import BaseModel, ConditionalInputsForDiscreteLayout
from .common.common import BaseDecoder, UserConstraintTransformerEncoder
from .common.image import ResnetFeatureExtractor
from .common.positional_encoding import build_position_encoding_2d
from .layoutformerpp.decoding_space_restriction import DECODE_SPACE_RESTRICTION
from .layoutformerpp.relation_restriction import TransformerSortByDictRelationConstraint
from .layoutformerpp.task_preprocessor import PREPROCESSOR
from .layoutformerpp.violate import calculate_vio_rate_relation, calculate_violation

logger = logging.getLogger(__name__)


class BaseAutoreg(BaseModel):
    """A simple regression model."""

    def __init__(
        self,
        features: ds.Features,
        tokenizer: LayoutSequenceTokenizer,
        d_model: int = 256,
        encoder_pos_emb: str = "sine",
        decoder_pos_emb: str = "layout",
        weight_init: bool = False,
        shared_embedding: bool = False,
        decoder_num_layers: int = 6,
        decoder_d_model: int = 256,
    ) -> None:
        super().__init__()
        self.features = features
        self.tokenizer = tokenizer
        self.d_model = d_model

        self.num_layers = num_layers = 6
        self.nhead = nhead = 8
        self.dim_feedforward = dim_feedforward = 4 * d_model
        logger.info(f"{dim_feedforward=}")

        self.encoder = ResnetFeatureExtractor(
            backbone="resnet50",
            d_model=d_model,
            head="transformer",
        )
        self.pos_emb_2d = build_position_encoding_2d(encoder_pos_emb, d_model)
        self.transformer_encoder = nn.TransformerEncoder(  # type: ignore
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

        self.decoder = BaseDecoder(
            d_label=self.tokenizer.N_total,
            d_model=decoder_d_model,
            num_layers=decoder_num_layers,
            nhead=nhead,
            pos_emb=decoder_pos_emb,
            dim_feedforward=dim_feedforward,
        )

        self.loss_fn_ce = nn.CrossEntropyLoss(
            label_smoothing=0.1, ignore_index=self.tokenizer.name_to_id("pad")
        )

        if weight_init:
            logger.info(f"Initialize weights of {self.__class__.__name__}")
            self.init_weights()

    def init_weights(self) -> None:
        self.decoder.init_weight()
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs: dict) -> dict[str, Tensor]:

        encoded_feat: dict[str, Tensor] = self._encode_into_memory(inputs)

        logits = self.decoder(
            tgt=inputs["seq"],
            tgt_key_padding_mask=inputs["tgt_key_padding_mask"],
            is_causal=True,
            **encoded_feat,
        )

        return {"logits": logits}

    def train_loss(
        self, inputs: dict, targets: dict, test: bool = False
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        outputs = self(inputs)
        logits = rearrange(outputs["logits"], "b s c -> b c s")
        nll_loss = self.loss_fn_ce(logits, targets["seq"])
        losses = {"nll_loss": nll_loss}
        return outputs, losses

    def sample(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: Optional[DictConfig] = None,
        cond_type: Optional[str] = "uncond",
        return_violation: bool = False,
        use_backtrack: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:  # type: ignore

        if cond_type == "relation" and use_backtrack:
            return self.sample_relation(
                cond=cond,
                batch_size=batch_size,
                sampling_cfg=sampling_cfg,
                return_violation=return_violation,
            )

        B = cond.image.size(0)
        if (B == 1) and batch_size and batch_size > 1:
            B = batch_size
            cond.image = repeat(cond.image, "1 x h w -> b x h w", b=B)

        token_mask = self.tokenizer.token_mask
        ids = self.special_token_ids

        input_ = torch.full((B, 1), fill_value=ids["bos"]).to(self.device)

        with torch.no_grad():
            encoder_input, seq_constraints = self._create_encoder_inputs(cond)
            encoded_feat = self._encode_into_memory(encoder_input)

        start_idx = 0
        if cond_type == "partial":
            input_ = torch.cat([input_, cond.seq[:, 1:6]], dim=1)
            start_idx = 5

        prepared_rel_constraints = []
        if cond_type == "relation":
            gen_r_constraint_fn = TransformerSortByDictRelationConstraint(
                self.preprocessor,
            )
            for batch_idx in range(B):
                rel_constraints = gen_r_constraint_fn.prepare(
                    seq_constraints["seq"][batch_idx]
                )
                prepared_rel_constraints.append(rel_constraints)

        for i in range(start_idx, self.tokenizer.max_token_length):

            with torch.no_grad():
                logits = self.decoder(
                    tgt=input_,
                    tgt_key_padding_mask=(input_ == ids["pad"]),
                    is_causal=True,
                    **encoded_feat,
                )

            # Decoding space restriction by just token mask
            logits = rearrange(logits[:, i : i + 1], "b 1 c -> b c")
            invalid = repeat(~token_mask[i : i + 1], "1 c -> b c", b=input_.size(0))
            logits[invalid] = -float("Inf")

            # Restrict a decoding space
            logits = DECODE_SPACE_RESTRICTION[cond_type](
                i + 1,
                cond.seq,
                logits,
                pad_id=ids["pad"],
                eos_id=ids["eos"],
                max_length=self.tokenizer.max_token_length,
            )

            predicted = sample(logits, sampling_cfg)
            input_ = torch.cat([input_, predicted], dim=1)

        torch.cuda.empty_cache()
        seq = input_[:, 1:].cpu()  # pop BOS
        output_seq: dict[str, Tensor] = self.postprocess({"seq": seq})

        if not return_violation:
            return output_seq

        violation_rate: dict[str, float] = calculate_violation(
            cond_type,
            cond,
            seq,
            output_seq,
            self.tokenizer,
            prepared_rel_constraints,
        )

        if cond_type in ["none", "uncond", "c", "cwh", "refinement"]:
            assert violation_rate["viorated"] == 0, f"{violation_rate=}"

        return output_seq, violation_rate

    def init_relation_count(self):
        return {
            "flag_idx": list(),
            "flag_idx": list(),
            "back_flag": False,
            "backtrack_count": 0,
        }

    @torch.no_grad()
    def sample_relation(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: Optional[DictConfig] = None,
        return_violation: bool = False,
        prob_gate: float = 0.3,
        RELATION_SIZE: int = 10,  # It means "10% of relation conds"
        **kwargs,
    ) -> dict[str, Tensor]:  # type: ignore

        logger.info(f"sample_relation with {RELATION_SIZE=}!")

        self.preprocessor.set_relation_size(RELATION_SIZE)

        B = cond.image.size(0)
        if (B == 1) and batch_size and batch_size > 1:
            B = batch_size
            cond.image = repeat(cond.image, "1 x h w -> b x h w", b=B)

        token_mask = self.tokenizer.token_mask
        ids = self.special_token_ids
        max_token_length = self.tokenizer.max_token_length + 2  # BOS, EOS

        # Encode image and constraints
        with torch.no_grad():
            encoder_input, seq_constraints = self._create_encoder_inputs(cond)
            encoded_feat = self._encode_into_memory(encoder_input)

        gen_r_constraint_fn = TransformerSortByDictRelationConstraint(
            self.preprocessor,
        )
        cond_chunked_seqs = torch.chunk(cond.seq, cond.seq.size(0))

        output = []
        prepared_rel_constraints = []

        for batch_idx in range(B):

            print(f"batch_idx: {batch_idx}")

            input_ = torch.full((1, 1), fill_value=ids["bos"]).to(self.device)
            __encoded_feat = {
                k: v[batch_idx].unsqueeze(0) for k, v in encoded_feat.items()
            }

            rel_constraints = gen_r_constraint_fn.prepare(
                seq_constraints["seq"][batch_idx]
            )
            REL_COUNT = self.init_relation_count()
            reset_num = 0
            idx = 0

            while idx < float("Inf"):

                with torch.no_grad():
                    logits = self.decoder(
                        tgt=input_,
                        tgt_key_padding_mask=(input_ == ids["pad"]),
                        is_causal=True,
                        **__encoded_feat,
                    )

                # Decoding space restriction by just token mask
                seq_len = logits.size(1) - 1
                logits = logits[:, -1]  # [b, c]
                invalid = repeat(
                    ~token_mask[seq_len : seq_len + 1], "1 c -> b c", b=input_.size(0)
                )
                logits[invalid] = -float("Inf")

                # Restrict a decoding space
                logits = DECODE_SPACE_RESTRICTION["relation"](
                    seq_len + 1,
                    cond_chunked_seqs[batch_idx],
                    logits,
                    pad_id=ids["pad"],
                    eos_id=ids["eos"],
                    max_length=self.tokenizer.max_token_length,
                )
                raw_logits = logits.clone()

                # Decoding space restriction
                mask, back_idx = gen_r_constraint_fn(input_, rel_constraints)
                mask = mask.unsqueeze(0)
                assert torch.all(~invalid[~mask]).item()
                logits[mask] = -float("Inf")

                # ProbabilityPruning
                pruned_logits = torch.where(
                    logits < prob_gate,
                    torch.full_like(logits, fill_value=-float("Inf")),
                    logits,
                )

                # Check whether the back
                if reset_num > 3:
                    logits = raw_logits.clone()
                    REL_COUNT["back_flag"] = False
                elif (
                    not REL_COUNT["back_flag"]
                    and REL_COUNT["flag_idx"].count(idx) < 5
                    and (pruned_logits.max() == -float("Inf")).item()
                    or (logits.max() == -float("Inf")).item()
                ):
                    REL_COUNT["flag_idx"].append(idx)
                    REL_COUNT["back_flag"] = True

                    if back_idx is not None:
                        idx = back_idx
                    else:
                        idx = random.randint(2, max(2, idx - 1))
                    input_ = input_[:, :idx]
                    REL_COUNT["backtrack_count"] += 1

                    if REL_COUNT["backtrack_count"] > 100:
                        reset_num += 1
                        REL_COUNT = self.init_relation_count()
                        input_ = torch.full((1, 1), fill_value=ids["bos"]).to(
                            self.device
                        )
                        idx = 0

                    continue

                if REL_COUNT["back_flag"]:
                    REL_COUNT["back_flag"] = False
                    temperature = 1.5
                else:
                    temperature = sampling_cfg.temperature

                predicted = sample(logits, sampling_cfg, temperature=temperature)
                input_ = torch.cat([input_, predicted], dim=1)

                if (
                    predicted.item() == ids["eos"]
                    or input_.size(1) == self.tokenizer.max_token_length + 1
                ):
                    break

                idx += 1

            input_ = torch.cat(
                [
                    input_,
                    torch.full(
                        (1, max_token_length - input_.size(1)), fill_value=ids["pad"]
                    ).type_as(input_),
                ],
                dim=1,
            )
            output.append(input_)
            prepared_rel_constraints.append(rel_constraints)
            torch.cuda.empty_cache()

        input_ = torch.cat(output, dim=0)
        seq = input_[:, 1:-1].cpu()  # pop BOS
        output_seq: dict[str, Tensor] = self.postprocess({"seq": seq})

        if not return_violation:
            return output_seq

        violation_rate: dict[str, float] = calculate_vio_rate_relation(
            cond,
            output_seq,
            prepared_rel_constraints,
        )
        print(f"{violation_rate=}")
        return output_seq, violation_rate

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:
        data = self.tokenizer.encode(inputs)
        image = torch.cat([inputs["image"], inputs["saliency"]], dim=1)
        inputs = {
            "seq": data["seq"][:, :-1],
            "tgt_key_padding_mask": ~data["mask"][:, :-1],
            "image": image,
        }
        targets = {"seq": data["seq"][:, 1:]}

        return inputs, targets

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        assert inputs["image"].size(1) == 4
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        # [bs, h*w, d_model]
        memory = self.transformer_encoder(input_img_feature)  # [bs, hw, d_model]
        return {"memory": memory}

    @abstractmethod
    def _create_encoder_inputs(
        self, cond: ConditionalInputsForDiscreteLayout
    ) -> dict[str, Tensor]:
        raise NotImplementedError


class Autoreg(BaseAutoreg):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _create_encoder_inputs(
        self, cond: ConditionalInputsForDiscreteLayout
    ) -> dict[str, Tensor]:
        return {"image": cond.image}


class BaseAuxilaryTaskAutoreg(BaseAutoreg):
    def __init__(
        self,
        auxilary_task: Optional[str] = None,
        use_flag_embedding: Optional[bool] = True,
        use_multitask: Optional[bool] = False,
        RELATION_SIZE: int = 10,
        shared_embedding: Optional[bool] = False,
        global_task_embedding: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert (
            auxilary_task in COND_TYPES
        ), f"{auxilary_task=} must be one of {COND_TYPES}"

        logger.info(
            f"Use {auxilary_task=}, {use_multitask=}, {shared_embedding=}, {global_task_embedding=}"
        )
        self.auxilary_task = auxilary_task
        self.use_multitask = use_multitask
        self.preprocessor = PREPROCESSOR[auxilary_task]

        if auxilary_task == "relation":
            self.preprocessor = functools.partial(
                self.preprocessor, RELATION_SIZE=RELATION_SIZE
            )
        self.global_task_embedding = global_task_embedding
        self.preprocessor = PREPROCESSOR[auxilary_task](
            tokenizer=self.tokenizer,
            global_task_embedding=global_task_embedding,
        )
        if shared_embedding:
            logger.info(f"Use shared embedding btw Transfomer Encoder and Decoder")
            embedding_layer = self.decoder.emb
        else:
            embedding_layer = None
        self.user_const_encoder = UserConstraintTransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            d_label=self.preprocessor.N_total,
            embedding_layer=embedding_layer,
            dim_feedforward=self.dim_feedforward,
        )
        self.user_const_encoder.init_weight()
        n_params = sum([p.numel() for p in self.user_const_encoder.parameters()]) / 1e6
        logger.info(f"[user_const_encoder] number of parameters: {n_params:.2f}M")

        self.use_flag_embedding = use_flag_embedding
        if use_flag_embedding:
            logger.info(f"{use_flag_embedding=}")
            NUM_EMB_CLASS = 2
            TARGET_EMB_CLASS = 1  # TODO: correct?
            self.task_emb = nn.Embedding(NUM_EMB_CLASS, TARGET_EMB_CLASS)
            nn.init.normal_(self.task_emb.weight, mean=0.0, std=0.02)
            self.register_buffer("flag_img", torch.zeros(1).long())
            self.register_buffer("flag_user_const", torch.ones(1).long())

    def set_task_preprocessor(self, task: str) -> None:
        """
        Setter for task preprocessor and task definition.
        """
        assert task in COND_TYPES, f"{task=} must be one of {COND_TYPES}"
        if not self.use_multitask:
            return
        # logger.info(f"Set task preprocessor from {self.auxilary_task} to {task}")
        self.auxilary_task = task
        self.preprocessor = PREPROCESSOR[self.auxilary_task](
            tokenizer=self.tokenizer,
            global_task_embedding=self.global_task_embedding,
        )

    def get_random_task(self) -> str:
        task_list: list[str] = [
            "uncond",
            "c",
            "cwh",
            "partial",
            "refinement",
            "relation",
        ]
        # Please see Tab. 3 in the supplementary material
        # in https://arxiv.org/pdf/2208.08037.pdf
        weight_list: list[float] = [1 / 12, 1 / 3, 1 / 3, 1 / 12, 1 / 3, 1 / 12]

        task: str = random.choices(task_list, weights=weight_list)[0]
        return task

    def sample(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: Optional[DictConfig] = None,
        **kwargs,
    ) -> dict[str, Tensor]:  # type: ignore
        """
        In test time, this function is called.
        """
        if self.use_multitask:
            current_task: str = cond.task
            self.set_task_preprocessor(current_task)
        return super().sample(cond, batch_size, sampling_cfg, **kwargs)

    def _create_encoder_inputs(
        self, cond: ConditionalInputsForDiscreteLayout
    ) -> dict[str, Tensor]:
        encoder_input = {"image": cond.image}
        seq_constraints = self.preprocessor(cond)
        encoder_input["seq_layout_const"] = seq_constraints["seq"]
        encoder_input["seq_layout_const_pad_mask"] = seq_constraints["pad_mask"]
        return encoder_input, seq_constraints

    def preprocess(self, inputs: dict) -> tuple[dict, dict]:
        """
        This function is called in the training loop.
        Args:
            inputs (dict): dict of input tensors from a dataloader
        Returns:
            _inputs (dict): dict of input tensors for the model
            _targets (dict): dict of target tensors for the model
        """
        if self.use_multitask:
            self.set_task_preprocessor(self.get_random_task())
        cond_inputs, inputs = get_condition(inputs, self.auxilary_task, self.tokenizer)
        seq_constraints = self.preprocessor(cond_inputs)

        _inputs, _targets = super().preprocess(inputs)
        _inputs["seq_layout_const"] = seq_constraints["seq"]
        _inputs["seq_layout_const_pad_mask"] = seq_constraints["pad_mask"]

        return _inputs, _targets


class SoftTokenAuxilaryTaskAutoreg(BaseAuxilaryTaskAutoreg):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = {}
        outputs["memory"] = super()._encode_into_memory(inputs)["memory"]

        encoded_const = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
        )

        outputs["soft_token"] = encoded_const
        outputs["soft_token_mask"] = inputs["seq_layout_const_pad_mask"]

        if self.use_flag_embedding:
            outputs["emb_soft_token"] = self.task_emb(self.flag_user_const)
            outputs["emb_decoder_token"] = self.task_emb(self.flag_img)

        return outputs


class ConcateAuxilaryTaskAutoreg(BaseAuxilaryTaskAutoreg):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        outputs = {}

        img_feature = super()._encode_into_memory(inputs)["memory"]

        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_feature.size(0)
            )
        else:
            task_token = None

        user_const_feature = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
            task_token=task_token,
        )

        if self.use_flag_embedding:
            emb_flag_img = self.task_emb(self.flag_img)
            img_feature = img_feature + emb_flag_img

            emb_flag_user_const = self.task_emb(self.flag_user_const)
            user_const_feature = user_const_feature + emb_flag_user_const

        outputs["memory"] = torch.cat([img_feature, user_const_feature], dim=1)

        return outputs
