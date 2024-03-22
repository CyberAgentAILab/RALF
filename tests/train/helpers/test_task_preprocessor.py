import copy
from dataclasses import asdict

import numpy as np
import torch
from image2layout.train.config import TokenizerConfig
from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from image2layout.train.helpers.task import get_condition
from image2layout.train.helpers.util import set_seed
from image2layout.train.models.layoutformerpp.relation_restriction import (
    TransformerSortByDictRelationConstraint,
)
from image2layout.train.models.layoutformerpp.task_preprocessor import (
    LabelPreprocessor,
    LabelSizePreprocessor,
    PartialPreprocessor,
    RefinementPreprocessor,
    RelationshipPreprocessor,
    UnconditionalPreprocessor,
)
from omegaconf import OmegaConf

set_seed(0)


def check_get_condition(cond_inputs):
    seq = cond_inputs.seq
    mask = cond_inputs.mask

    valid_tokens = seq[mask].tolist()
    assert not -1 in valid_tokens

    invalid_tokens = seq[~mask].tolist()
    invalid_tokens = list(set(invalid_tokens))

    assert (
        len(invalid_tokens) == 0 or len(invalid_tokens) == 1
    ), f"invalid_tokens: {invalid_tokens}"


def check_output(seq, preprocessor):
    assert preprocessor.N_total > 0.0
    assert seq["seq"].min() >= 0.0
    assert seq["seq"].max() <= preprocessor.N_total

    pad_mask = seq["pad_mask"]
    seq = seq["seq"]

    assert len(list(set(seq[pad_mask].tolist()))) == 0 or list(
        set(seq[pad_mask].tolist())
    ) == [preprocessor.name_to_id("pad")]
    assert not preprocessor.name_to_id("pad") in list(set(seq[~pad_mask].tolist()))

    emb = torch.nn.Embedding(preprocessor.N_total, 256)
    _h = emb(seq)
    assert isinstance(_h, torch.Tensor)


def test_unconditional_preprocess(batch, layout_tokenizer, global_task_embedding):
    cond_inputs, _ = get_condition(batch, "none", layout_tokenizer)
    assert cond_inputs.seq is None and cond_inputs.mask is None
    preprocessor = UnconditionalPreprocessor(
        tokenizer=layout_tokenizer,
        global_task_embedding=global_task_embedding,
    )
    seq = preprocessor(cond_inputs)
    check_output(seq, preprocessor)


def test_label_preprocess(batch, layout_tokenizer, global_task_embedding):

    cond_inputs, _ = get_condition(batch, "c", layout_tokenizer)
    check_get_condition(cond_inputs)
    preprocessor = LabelPreprocessor(
        tokenizer=layout_tokenizer,
        global_task_embedding=global_task_embedding,
    )
    seq = preprocessor(cond_inputs)
    check_output(seq, preprocessor)


def test_label_size_preprocess(batch, layout_tokenizer, global_task_embedding):
    cond_inputs, _ = get_condition(batch, "cwh", layout_tokenizer)
    check_get_condition(cond_inputs)
    preprocessor = LabelSizePreprocessor(
        tokenizer=layout_tokenizer,
        global_task_embedding=global_task_embedding,
    )
    seq = preprocessor(cond_inputs)
    check_output(seq, preprocessor)


def test_refinement_preprocess(batch, layout_tokenizer, global_task_embedding):
    ori_cond = layout_tokenizer.encode(copy.deepcopy(batch))
    cond_inputs, _ = get_condition(batch, "refinement", layout_tokenizer)
    check_get_condition(cond_inputs)
    new_cond = layout_tokenizer.encode(batch)  # Perturbed condition
    preprocessor = RefinementPreprocessor(
        tokenizer=layout_tokenizer,
        global_task_embedding=global_task_embedding,
    )
    seq = preprocessor(cond_inputs)
    check_output(seq, preprocessor)


def test_partial_preprocess(batch, layout_tokenizer, global_task_embedding):
    cond_inputs, _ = get_condition(batch, "partial", layout_tokenizer)
    check_get_condition(cond_inputs)
    preprocessor = PartialPreprocessor(
        tokenizer=layout_tokenizer,
        global_task_embedding=global_task_embedding,
    )
    seq = preprocessor(cond_inputs)
    check_output(seq, preprocessor)


def test_relation_preprocess(batch, layout_tokenizer, global_task_embedding):
    cond_inputs, _ = get_condition(batch, "relation", layout_tokenizer)
    check_get_condition(cond_inputs)
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer,
        global_task_embedding=global_task_embedding,
    )
    seq = preprocessor(cond_inputs)
    check_output(seq, preprocessor)

    pad_mask = seq["pad_mask"]
    seq = seq["seq"]

    for i in range(seq.size(0)):

        _seq = seq[i]

        gen_r_constraint_fn = TransformerSortByDictRelationConstraint(
            preprocessor,
        )

        gen_r_constraint_fn.prepare(_seq)


def _decode_seq(seq, preprocessor):

    out = []
    for _seq in seq:
        _tmp = []
        for e in _seq:
            _tmp.append(preprocessor.id_to_name(e.item()))
        out.append(_tmp)
    return np.array(out)


if __name__ == "__main__":

    features = torch.load("tmp/features.pt")
    batch = torch.load("tmp/batch.pt")
    batch = {k: v[:4] for k, v in batch.items()}

    train_cfg = OmegaConf.create(
        {
            "tokenizer": asdict(TokenizerConfig()),
        }
    )

    layout_tokenizer = LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=10,
        **OmegaConf.to_container(train_cfg.tokenizer),
    )

    # for global_task_embedding in [True, False]:
    for global_task_embedding in [False]:
        print(f"{global_task_embedding=}")
        test_unconditional_preprocess(batch, layout_tokenizer, global_task_embedding)
        test_label_preprocess(batch, layout_tokenizer, global_task_embedding)
        test_label_size_preprocess(batch, layout_tokenizer, global_task_embedding)
        test_refinement_preprocess(batch, layout_tokenizer, global_task_embedding)
        test_partial_preprocess(batch, layout_tokenizer, global_task_embedding)
        test_relation_preprocess(batch, layout_tokenizer, global_task_embedding)

    # OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 poetry run python -m tests.train.helpers.test_task_preprocessor
