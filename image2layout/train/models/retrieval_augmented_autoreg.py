import functools
import logging
import random
from abc import abstractmethod
from typing import Optional, Union

import datasets as ds
import torch
import torch.nn as nn
from einops import rearrange, repeat
from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from image2layout.train.helpers.sampling import sample
from omegaconf import DictConfig
from torch import Tensor

from ..fid.model import load_fidnet_feature_extractor
from ..helpers.task import COND_TYPES, get_condition
from .common.attention import Attention, FeedForward
from .common.base_model import (
    BaseModel,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)
from .common.common import BaseDecoder, UserConstraintTransformerEncoder
from .common.image import ResnetFeatureExtractor
from .common.positional_encoding import (
    build_position_encoding_1d,
    build_position_encoding_2d,
)
from .layoutformerpp.decoding_space_restriction import DECODE_SPACE_RESTRICTION
from .layoutformerpp.relation_restriction import TransformerSortByDictRelationConstraint
from .layoutformerpp.task_preprocessor import PREPROCESSOR
from .layoutformerpp.violate import calculate_vio_rate_relation, calculate_violation

logger = logging.getLogger(__name__)


def get_ref_layout_input(
    retrieved_samples: dict,
    kdx: int,
) -> dict:
    return {
        # [B, N, max_elem] -> [B, max_elem]
        "center_x": retrieved_samples["center_x"][:, kdx],
        # [B, N, max_elem] -> [B, max_elem]
        "center_y": retrieved_samples["center_y"][:, kdx],
        "width": retrieved_samples["width"][
            :, kdx
        ],  # [B, N, max_elem] -> [B, max_elem]
        "height": retrieved_samples["height"][
            :, kdx
        ],  # [B, N, max_elem] -> [B, max_elem]
        # [B, N, max_elem] -> [B, max_elem]
        "label": retrieved_samples["label"][:, kdx].long(),
        "mask": retrieved_samples["mask"][
            :, kdx
        ].bool(),  # [B, N, max_elem] -> [B, max_elem]
    }


class BaseRetrievalAugmentedAutoreg(BaseModel):
    def __init__(
        self,
        features: ds.Features,
        tokenizer: LayoutSequenceTokenizer,
        dataset_name: str,
        max_seq_length: int,
        db_dataset: ds.Dataset,
        d_model: int = 256,
        encoder_pos_emb: str = "sine",
        decoder_pos_emb: str = "layout",
        weight_init: bool = True,
        top_k: int = 16,
        layout_backbone: str = "feature_extractor",
        use_reference_image: bool = False,
        freeze_layout_encoder: bool = True,
        retrieval_backbone: str = "saliency",
        random_retrieval: bool = False,
        saliency_k: Union[int, str] = 8,
        decoder_d_model: int = 256,
    ):
        super(BaseRetrievalAugmentedAutoreg, self).__init__()

        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.features = features
        self.use_reference_image = use_reference_image
        self.layout_backbone = layout_backbone
        self.top_k = top_k
        self.weight_init = weight_init
        self.retrieval_backbone = retrieval_backbone
        self.random_retrieval = random_retrieval

        self.saliency_k = saliency_k
        logger.info(f"{self.saliency_k=}")
        if self.saliency_k == "dynamic":
            NUM_RETRIEVAL_METHODS = 2
            TARGET_EMB_CLASS = 1  # TODO: correct?
            self.emb_hybrid_ret = nn.Embedding(NUM_RETRIEVAL_METHODS, TARGET_EMB_CLASS)
            nn.init.normal_(self.emb_hybrid_ret.weight, mean=0.0, std=0.02)

        self.num_layers = num_layers = 6
        self.nhead = nhead = 8
        self.dropout = dropout = 0.1
        self.encoder = ResnetFeatureExtractor(
            backbone="resnet50", d_model=d_model, head="transformer"
        )
        self.compute_params(self.encoder, "Image encoder (ResNet50)")
        self.pos_emb_2d = build_position_encoding_2d(encoder_pos_emb, d_model)

        self.dim_feedforward = dim_feedforward = 4 * d_model
        logger.info(f"{dim_feedforward=}")

        # For image feature from ResNet50
        self.transformer_encoder = nn.TransformerEncoder(  # type: ignore
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=dropout,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )
        self.compute_params(
            self.transformer_encoder, "Image encoder (Transformer Encoder)"
        )

        self.decoder = BaseDecoder(
            d_label=self.tokenizer.N_total,
            d_model=decoder_d_model,
            num_layers=num_layers,
            nhead=nhead,
            pos_emb=decoder_pos_emb,
            dim_feedforward=dim_feedforward,
        )
        self.compute_params(self.decoder, "Layout decoder (Transformer Decoder)")
        self.loss_fn_ce = nn.CrossEntropyLoss(
            label_smoothing=0.1, ignore_index=self.tokenizer.name_to_id("pad")
        )

        self.layout_encoer = load_fidnet_feature_extractor(
            dataset_name=dataset_name,
            num_classes=features["label"].feature.num_classes,
            max_seq_length=max_seq_length,
        )
        logger.info(f"Build a layout feature extractor ({freeze_layout_encoder=})")
        self.layout_encoer.enc_transformer.token.requires_grad = False
        assert freeze_layout_encoder is True

        for p in self.layout_encoer.parameters():
            p.requires_grad = False
        self.compute_params(self.layout_encoer, "Layout Encoder")

        self.pos_emb_1d = build_position_encoding_1d(
            pos_emb="layout",
            d_model=d_model,
            max_len=5000 if not self.use_reference_image else 10000,
        )
        self.layout_adapter = FeedForward(
            dim=256, hidden_dim=4 * self.d_model, output_dim=self.d_model
        )
        self.head = FeedForward(
            dim=self.d_model, hidden_dim=4 * self.d_model, dropout=0.0
        )
        self.compute_params(self.layout_adapter, "Layout Adapter")
        self.compute_params(self.head, "Layout Adapter Head")

    def init_weights(self) -> None:
        self.decoder.init_weight()
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @abstractmethod
    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def _create_encoder_inputs(
        self, cond: RetrievalAugmentedConditionalInputsForDiscreteLayout
    ) -> dict[str, Tensor]:
        encoder_input = {
            "image": cond.image,
            "retrieved": cond.retrieved,
        }
        return encoder_input, None

    def forward(self, inputs: dict) -> dict[str, Tensor]:

        inputs["retrieved"] = {
            k: v.type_as(inputs["image"])
            for k, v in inputs["retrieved"].items()
            if isinstance(v, Tensor)
        }

        encoded_feat: dict[str, Tensor] = self._encode_into_memory(inputs)

        # Transformer decoder
        logits = self.decoder(
            tgt=inputs["seq"],  # [bs, 5*max_elem]
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
        cond: RetrievalAugmentedConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: Optional[DictConfig] = None,
        cond_type: Optional[str] = "uncond",
        return_violation: bool = False,
        use_backtrack: bool = True,
        return_decoded_cond: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:  # type: ignore

        if cond_type == "relation" and use_backtrack:
            return self.sample_relation(
                cond=cond,
                batch_size=batch_size,
                sampling_cfg=sampling_cfg,
                cond_type=cond_type,
                return_violation=return_violation,
                return_decoded_cond=return_decoded_cond,
                **kwargs,
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

        if return_decoded_cond:
            decoded_tokens = self.preprocessor.decode_tokens(
                encoder_input["seq_layout_const"]
            )
            output_seq["decoded_tokens"] = decoded_tokens

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
        # print(f"{violation_rate=}")

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
        cond: RetrievalAugmentedConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: Optional[DictConfig] = None,
        return_violation: bool = False,
        prob_gate: float = 0.3,
        RELATION_SIZE: int = 10,  # It means "10% of relation conds"
        return_decoded_cond: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:  # type: ignore

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
                    and (pruned_logits.max() == -float("Inf")).item()
                    or (logits.max() == -float("Inf")).item()
                ):
                    REL_COUNT["flag_idx"].append(idx)
                    REL_COUNT["back_flag"] = True

                    if back_idx is not None and REL_COUNT["flag_idx"].count(idx) < 3:
                        idx = back_idx
                    else:
                        idx = random.randint(2, max(2, idx - 1))
                    input_ = input_[:, :idx]
                    REL_COUNT["backtrack_count"] += 1

                    if REL_COUNT["backtrack_count"] > 30:

                        REL_COUNT = self.init_relation_count()
                        reset_num += 1
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
                        (1, max_token_length - input_.size(1)), fill_value=True
                    ).type_as(input_),
                ],
                dim=1,
            )
            output.append(input_)
            prepared_rel_constraints.append(rel_constraints)
            torch.cuda.empty_cache()

        input_ = torch.cat(output, dim=0)
        seq = input_[:, 1:-1].cpu()  # pop BOS, EOS
        output_seq: dict[str, Tensor] = self.postprocess({"seq": seq})

        if return_decoded_cond:
            decoded_tokens = self.preprocessor.decode_tokens(
                encoder_input["seq_layout_const"]
            )
            output_seq["decoded_tokens"] = decoded_tokens

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
        assert (
            inputs["retrieved"]["image"].size(2) == 4
        ), f"{inputs['retrieved']['image'].shape=}"
        inputs = {
            "seq": data["seq"][:, :-1],
            "tgt_key_padding_mask": ~data["mask"][:, :-1],
            "image": image,
            "retrieved": inputs["retrieved"],
        }

        targets = {"seq": data["seq"][:, 1:]}
        return inputs, targets


def extract_retrieved_features(
    *,
    retrieved_samples,
    top_k,
    image_encoder,
    layout_encoder,
    layout_adapter,
    pos_emb_1d,
    use_reference_image,
):
    ref_images = []
    ref_layouts = []

    for kdx in range(top_k):

        # Encode image if use_reference_image is True
        if use_reference_image:
            # Encode image
            with torch.no_grad():
                # feature_img_ref = image_encoder(
                #     retrieved_samples["image"][:, kdx]
                # )  # [bs, d_model, h, w]
                # feature_img_ref = nn.AdaptiveAvgPool2d((1, 1))(
                #     feature_img_ref
                # )  # [bs, d_model, 1, 1]
                # feature_img_ref = rearrange(feature_img_ref, "b c 1 1 -> b c")
                feature_img_ref = image_encoder[2](
                    image_encoder[1](
                        image_encoder[0](retrieved_samples["image"][:, kdx])
                    )
                )
                ref_images.append(feature_img_ref.detach())  # To avoid OOM error

        # Encode layout
        ref_layout_input = get_ref_layout_input(retrieved_samples, kdx)
        # Encode layout [B, max_elem] -> [B, d_model]. One layout corresponds to one feature.
        with torch.no_grad():
            feature_layout_ref = layout_encoder.extract_features(
                ref_layout_input
            )  # [B, d_model]
        # No positional encoding because the layout has no sequence, just feature.
        feature_layout_ref = layout_adapter(feature_layout_ref)
        ref_layouts.append(feature_layout_ref)

    if use_reference_image:
        ref_images = torch.cat(ref_images, dim=1)
    ref_layouts = torch.stack(ref_layouts, dim=1)  # [bs, top_k, d_model]
    assert ref_layouts.size(1) == top_k
    if use_reference_image:
        ref_layouts = torch.cat(
            [ref_layouts, ref_images], dim=1
        )  # [bs, top_k, 2*d_model]

    if "hybrid_dynamic_indexes" in retrieved_samples.keys():
        ref_layouts = ref_layouts + retrieved_samples["hybrid_dynamic_indexes"]

    ref_layouts = pos_emb_1d(ref_layouts)  # [bs, top_k, d_model]

    return ref_layouts


class RetrievalAugmentedAutoregAdapter(BaseRetrievalAugmentedAutoreg):
    """
    Use a cross-attention layer to fuse image and retrieved layout features.
    Unique point is in `# 3. Transformer encoder to create memory`
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        self.attn = Attention(
            self.d_model, self.d_model, heads=8, dim_head=64, dropout=0.0
        )
        self.init_weights()

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        # 1. Encode input image
        assert inputs["image"].size(1) == 4
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        # [bs, h*w, d_model]
        memory = self.transformer_encoder(input_img_feature)  # [bs, hw, d_model]

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=inputs["retrieved"],
            top_k=self.top_k,
            image_encoder=[self.encoder, self.pos_emb_2d, self.transformer_encoder],
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )

        # 3. Transformer encoder to create memory
        memory = self.attn(memory, ref_layouts)  #######
        memory = self.head(memory)

        return {"memory": memory}


class BaseAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg(BaseRetrievalAugmentedAutoreg):
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
            dim_token = self.preprocessor.N_total + self.tokenizer.N_total
            self.decoder.reset_embedding_layer(dim_token)
            embedding_layer = self.decoder.emb  # ass by reference
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
        self.compute_params(
            self.user_const_encoder, "Constraint Encoder (Transformer Decoder)"
        )

        self.use_flag_embedding = use_flag_embedding
        if use_flag_embedding:
            logger.info(f"{use_flag_embedding=}")
            NUM_EMB_CLASS = 2
            TARGET_EMB_CLASS = 1  # TODO: correct?
            self.task_emb = nn.Embedding(NUM_EMB_CLASS, TARGET_EMB_CLASS)
            nn.init.normal_(self.task_emb.weight, mean=0.0, std=0.02)
            self.register_buffer("flag_img", torch.zeros(1).long())
            self.register_buffer("flag_user_const", torch.ones(1).long())

        # Fusion layer
        self.attn = Attention(
            self.d_model, self.d_model, heads=8, dim_head=64, dropout=0.0
        )
        self.init_weights()
        self.compute_params(self.attn, "Modalitity Fuser (Cross-attention)")

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
        cond: RetrievalAugmentedConditionalInputsForDiscreteLayout,
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
        self, cond: RetrievalAugmentedConditionalInputsForDiscreteLayout
    ) -> dict[str, Tensor]:
        encoder_input = {
            "image": cond.image,
            "retrieved": cond.retrieved,
        }
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
        # Please call get_condition before super().preprocess,
        # because, in get_condition, inputs are sorted.
        if self.use_multitask:
            self.set_task_preprocessor(self.get_random_task())
        cond_inputs, inputs = get_condition(inputs, self.auxilary_task, self.tokenizer)
        seq_constraints = self.preprocessor(cond_inputs)

        _inputs, _targets = super().preprocess(inputs)

        _inputs["seq_layout_const"] = seq_constraints["seq"]
        _inputs["seq_layout_const_pad_mask"] = seq_constraints["pad_mask"]

        return _inputs, _targets

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # 1. Encode input image
        assert inputs["image"].size(1) == 4, f"{inputs['image'].shape=}"
        assert (
            inputs["retrieved"]["image"].size(2) == 4
        ), f"{inputs['retrieved']['image'].shape=}"
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        # [bs, h*w, d_model]
        memory = self.transformer_encoder(input_img_feature)  # [bs, hw, d_model]

        if self.saliency_k == "dynamic":
            inputs["retrieved"]["hybrid_dynamic_indexes"] = self.emb_hybrid_ret(
                inputs["retrieved"]["hybrid_dynamic_indexes"].long()
            )  # [bs, 16, 1]

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=inputs["retrieved"],
            top_k=self.top_k,
            image_encoder=[self.encoder, self.pos_emb_2d, self.transformer_encoder],
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )

        # 3. Transformer encoder to create memory
        memory = self.attn(memory, ref_layouts)  #### (Optional: Cross-attn or Concat)
        memory = self.head(memory)
        return {"memory": memory}


class ConcateAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        img_retrieved_layout_memory = super()._encode_into_memory(inputs)["memory"]

        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_retrieved_layout_memory.size(0)
            ).type_as(inputs["seq_layout_const"])
        else:
            task_token = None

        # 4. Encode user-specific layout constraints
        user_const_feature = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
            task_token=task_token,
        )

        # 5. (Optional) Add flag embedding
        if self.use_flag_embedding:
            img_retrieved_layout_memory = img_retrieved_layout_memory + self.task_emb(
                self.flag_img
            )
            user_const_feature = user_const_feature + self.task_emb(
                self.flag_user_const
            )

        # 6. Concat memory and user-specific layout constraints
        memory = torch.cat([img_retrieved_layout_memory, user_const_feature], dim=1)

        return {"memory": memory}


class BaseAuxilaryTaskConcatRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg
):
    """
    Use a concatenation to fuse image and retrieved layout features.
    Unique point is in `# 3. Transformer encoder to create memory`
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.init_weights()

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        # 1. Encode input image
        assert inputs["image"].size(1) == 4
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        # [bs, h*w, d_model]
        memory = self.transformer_encoder(input_img_feature)  # [bs, hw, d_model]

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=inputs["retrieved"],
            top_k=self.top_k,
            image_encoder=[self.encoder, self.pos_emb_2d, self.transformer_encoder],
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )

        # 3. Transformer encoder to create memory
        # [bs, hw+k, d_model]
        memory = self.head(
            torch.cat([memory, ref_layouts], dim=1)
        )  # [bs, hw+k, d_model]

        return {"memory": memory}


class ConcateAuxilaryTaskConcateRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskConcatRetrievalAugmentedAutoreg
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        img_retrieved_layout_memory = super()._encode_into_memory(inputs)["memory"]

        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_retrieved_layout_memory.size(0)
            ).type_as(inputs["seq_layout_const"])
        else:
            task_token = None

        # 4. Encode user-specific layout constraints
        user_const_feature = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
            task_token=task_token,
        )

        # 5. (Optional) Add flag embedding
        if self.use_flag_embedding:
            img_retrieved_layout_memory = img_retrieved_layout_memory + self.task_emb(
                self.flag_img
            )
            user_const_feature = user_const_feature + self.task_emb(
                self.flag_user_const
            )

        # 6. Concat memory and user-specific layout constraints
        memory = torch.cat([img_retrieved_layout_memory, user_const_feature], dim=1)

        return {"memory": memory}


class BaseAuxilaryTaskConcatCrossAttnRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg
):
    """
    Use a concatenation to fuse image and retrieved layout features.
    Unique point is in `# 3. Transformer encoder to create memory`
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.init_weights()

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        # 1. Encode input image
        assert inputs["image"].size(1) == 4
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        # [bs, h*w, d_model]
        memory = self.transformer_encoder(input_img_feature)  # [bs, hw, d_model]

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=inputs["retrieved"],
            top_k=self.top_k,
            image_encoder=[self.encoder, self.pos_emb_2d, self.transformer_encoder],
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )

        # 3. Transformer encoder to create memory
        # [bs, hw+k, d_model]
        #### Cross-attn
        memory_ca = self.attn(
            memory, ref_layouts
        )  #### (Optional: Cross-attn or Concat)

        memory = self.head(
            torch.cat([memory, memory_ca, ref_layouts], dim=1)
        )  # [bs, hw+k, d_model]

        return {"memory": memory}


# Final architecture
class ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskConcatCrossAttnRetrievalAugmentedAutoreg
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        img_retrieved_layout_memory = super()._encode_into_memory(inputs)["memory"]

        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_retrieved_layout_memory.size(0)
            ).type_as(inputs["seq_layout_const"])
        else:
            task_token = None

        # 4. Encode user-specific layout constraints
        user_const_feature = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
            task_token=task_token,
        )

        # 5. (Optional) Add flag embedding
        if self.use_flag_embedding:
            img_retrieved_layout_memory = img_retrieved_layout_memory + self.task_emb(
                self.flag_img
            )
            user_const_feature = user_const_feature + self.task_emb(
                self.flag_user_const
            )

        # 6. Concat memory and user-specific layout constraints
        memory = torch.cat([img_retrieved_layout_memory, user_const_feature], dim=1)

        return {"memory": memory}


#


class BaseAuxilaryTaskFlagConcatCrossAttnRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg
):
    """
    Use a concatenation to fuse image and retrieved layout features.
    Unique point is in `# 3. Transformer encoder to create memory`
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.init_weights()

        NUM_EMB_CLASS = 2
        TARGET_EMB_CLASS = 1  # TODO: correct?
        self.img_or_layout_emb = nn.Embedding(NUM_EMB_CLASS, TARGET_EMB_CLASS)
        nn.init.normal_(self.img_or_layout_emb.weight, mean=0.0, std=0.02)
        self.register_buffer("flag_img", torch.zeros(1).long())
        self.register_buffer("flag_layout", torch.ones(1).long())

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        # 1. Encode input image
        assert inputs["image"].size(1) == 4
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        # [bs, h*w, d_model]
        memory = self.transformer_encoder(input_img_feature)  # [bs, hw, d_model]

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=inputs["retrieved"],
            top_k=self.top_k,
            image_encoder=[self.encoder, self.pos_emb_2d, self.transformer_encoder],
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )

        # 3. Transformer encoder to create memory
        # [bs, hw+k, d_model]
        memory = memory + self.img_or_layout_emb(self.flag_img)
        ref_layouts = ref_layouts + self.img_or_layout_emb(self.flag_layout)

        #### Cross-attn
        memory_ca = self.attn(
            memory, ref_layouts
        )  #### (Optional: Cross-attn or Concat)

        memory = self.head(
            torch.cat([memory, memory_ca, ref_layouts], dim=1)
        )  # [bs, hw+k, d_model]

        return {"memory": memory}


class ConcateAuxilaryTaskFlagConcateCrossAttnRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskFlagConcatCrossAttnRetrievalAugmentedAutoreg
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        img_retrieved_layout_memory = super()._encode_into_memory(inputs)["memory"]

        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_retrieved_layout_memory.size(0)
            ).type_as(inputs["seq_layout_const"])
        else:
            task_token = None

        # 4. Encode user-specific layout constraints
        user_const_feature = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
            task_token=task_token,
        )

        # 5. (Optional) Add flag embedding
        if self.use_flag_embedding:
            img_retrieved_layout_memory = img_retrieved_layout_memory + self.task_emb(
                self.flag_img
            )
            user_const_feature = user_const_feature + self.task_emb(
                self.flag_user_const
            )

        # 6. Concat memory and user-specific layout constraints
        memory = torch.cat([img_retrieved_layout_memory, user_const_feature], dim=1)

        return {"memory": memory}


class BaseAuxilaryTaskConcatTransEncRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg
):
    """
    Use a concatenation to fuse image and retrieved layout features.
    Unique point is in `# 3. Transformer encoder to create memory`
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.init_weights()

        NUM_EMB_CLASS = 2
        TARGET_EMB_CLASS = 1  # TODO: correct?
        self.img_or_layout_emb = nn.Embedding(NUM_EMB_CLASS, TARGET_EMB_CLASS)
        nn.init.normal_(self.img_or_layout_emb.weight, mean=0.0, std=0.02)
        self.register_buffer("flag_img", torch.zeros(1).long())
        self.register_buffer("flag_layout", torch.ones(1).long())

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        # 1. Encode input image
        assert inputs["image"].size(1) == 4
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        # [bs, h*w, d_model]

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=inputs["retrieved"],
            top_k=self.top_k,
            image_encoder=[self.encoder, self.pos_emb_2d],
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )

        # 3. Transformer encoder to create memory
        memory_ca = self.attn(
            input_img_feature, ref_layouts
        )  #### (Optional: Cross-attn or Concat)
        feat = torch.cat(
            [input_img_feature, memory_ca, ref_layouts], dim=1
        )  # [bs, hw+k, d_model]
        memory = self.transformer_encoder(feat)  # [bs, hw, d_model]

        return {"memory": memory}


class ConcateAuxilaryTaskConcateTransEncRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskConcatTransEncRetrievalAugmentedAutoreg
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        img_retrieved_layout_memory = super()._encode_into_memory(inputs)["memory"]

        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_retrieved_layout_memory.size(0)
            ).type_as(inputs["seq_layout_const"])
        else:
            task_token = None

        # 4. Encode user-specific layout constraints
        user_const_feature = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
            task_token=task_token,
        )

        # 5. (Optional) Add flag embedding
        if self.use_flag_embedding:
            img_retrieved_layout_memory = img_retrieved_layout_memory + self.task_emb(
                self.flag_img
            )
            user_const_feature = user_const_feature + self.task_emb(
                self.flag_user_const
            )

        # 6. Concat memory and user-specific layout constraints
        memory = torch.cat([img_retrieved_layout_memory, user_const_feature], dim=1)

        return {"memory": memory}


class BaseAuxilaryTaskAfterConcatTransEncRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg
):
    """
    Use a concatenation to fuse image and retrieved layout features.
    Unique point is in `# 3. Transformer encoder to create memory`
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.init_weights()

        self.encoder_modality = nn.TransformerEncoder(  # type: ignore
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
                dim_feedforward=self.dim_feedforward,
            ),
            num_layers=self.num_layers,
        )
        for p in self.encoder_modality.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        # 1. Encode input image
        assert inputs["image"].size(1) == 4
        input_img_feature = self.encoder(inputs["image"])  # [bs, d_model, h, w]
        input_img_feature = self.pos_emb_2d(input_img_feature)
        memory = self.transformer_encoder(input_img_feature)  # [bs, hw, d_model]
        # [bs, h*w, d_model]

        # 2. Encode retrieved images and layout.
        ref_layouts = extract_retrieved_features(
            retrieved_samples=inputs["retrieved"],
            top_k=self.top_k,
            image_encoder=[self.encoder, self.pos_emb_2d],
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
            use_reference_image=self.use_reference_image,
        )

        # 3. Transformer encoder to create memory
        memory = torch.cat([memory, ref_layouts], dim=1)  # [bs, hw+k, d_model]

        memory = self.encoder_modality(memory)

        return {"memory": memory}


class ConcateAuxilaryTaskAfterConcateTransEncRetrievalAugmentedAutoreg(
    BaseAuxilaryTaskAfterConcatTransEncRetrievalAugmentedAutoreg
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _encode_into_memory(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        img_retrieved_layout_memory = super()._encode_into_memory(inputs)["memory"]

        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_retrieved_layout_memory.size(0)
            ).type_as(inputs["seq_layout_const"])
        else:
            task_token = None

        # 4. Encode user-specific layout constraints
        user_const_feature = self.user_const_encoder(
            src=inputs["seq_layout_const"],
            src_key_padding_mask=inputs["seq_layout_const_pad_mask"],
            task_token=task_token,
        )

        # 5. (Optional) Add flag embedding
        if self.use_flag_embedding:
            img_retrieved_layout_memory = img_retrieved_layout_memory + self.task_emb(
                self.flag_img
            )
            user_const_feature = user_const_feature + self.task_emb(
                self.flag_user_const
            )

        # 6. Concat memory and user-specific layout constraints
        memory = torch.cat([img_retrieved_layout_memory, user_const_feature], dim=1)

        return {"memory": memory}
