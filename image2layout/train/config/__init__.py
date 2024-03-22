"""
A file to declare dataclass instances used for hydra configs at ./config/*
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

from hydra.conf import RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .data import DataConfig
from .dataset import dataset_config_factory, dataset_config_names
from .sampling import sampling_config_factory, sampling_config_names
from .tokenizer import TokenizerConfig


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    freeze_lr_epoch: int = 50
    freeze_dis_epoch: int = 50
    warmup_dis_epoch: int = 100
    plot_scalars_interval: int = 10
    plot_generated_samples_epoch_interval: int = 5
    log_level: str = "info"
    save_tmp_model_epoch: int = 10000000
    save_vis_epoch: int = 100000
    clip_max_norm: float = 1.0
    num_workers: int = 4

    num_trainset: Optional[int] = None


def init_train_config_store() -> ConfigStore:
    cs = ConfigStore.instance()
    cs.store(group="data", name="config_data_default", node=DataConfig)
    cs.store(group="tokenizer", name="config_tokenizer_default", node=TokenizerConfig)
    cs.store(group="training", name="config_training_default", node=TrainingConfig)

    for name in sampling_config_names():
        cs.store(group="sampling", name=name, node=sampling_config_factory(name))
    for name in dataset_config_names():
        cs.store(group="dataset", name=name, node=dataset_config_factory(name))
    return cs


@dataclass
class TestConfig:
    job_dir: str
    result_dir: str
    sampling: Any
    cond_type: Optional[str] = None  # See COND_TYPES in task.py
    batch_size: int = 128
    dataset_path: Optional[str] = None
    # target_split: str = "test"  # for evaluation. ["train", "val", "test", "eval"]
    debug: bool = False  # disable some features to enable fast runtime
    debug_num_samples: int = -1  # in debug mode, reduce the number of samples when > 0
    best_or_final: str = "final"
    num_seeds: int = 3  # number of seeds to average
    ckpt_filter_substring: Optional[str] = None  # filter ckpt by substring in path
    test_split: str = "test"  # for evaluation. ["val", "test"]
    preload_data: bool = True  # set True if only w/ enough memory and want to speedup

    # for retrieval-augmented models
    use_db_dataset: bool = False  # For ramdom-retrieval-augmented autoreg

    # for diffusion models, refinement only
    refine_lambda: float = 3.0  # if > 0.0, trigger refinement mode
    refine_mode: str = "uniform"
    refine_offset_ratio: float = 0.1  # 0.2

    # for diffusion models, relation only
    relation_lambda: float = 3e6  # if > 0.0, trigger relation mode
    relation_num_update: int = 3

    # For our hybrid retrieval
    inference_num_saliency: int = 8

    # For inference_single_data.py
    sample_id: Optional[Union[int, str]] = None

    # For interence_no_anno_data.py to change dataset
    no_anno_dataset_name: str = "pku10"

    dynamic_topk: Optional[int] = None

    save_vis: bool = False

    repeat_retrieved_layouts: bool = False


@dataclass
class MyRunDir(RunDir):  # type: ignore
    dir: str = "/tmp/hydra-runs/inference/run/${now:%Y-%m-%d_%H-%M-%S}"


@dataclass
class MySweepDir(RunDir):  # type: ignore
    dir: str = "/tmp/hydra-runs/inference/sweep/${now:%Y-%m-%d_%H-%M-%S}"


def init_test_config_store() -> ConfigStore:
    cs = ConfigStore.instance()
    cs.store(name="test_config", node=TestConfig)
    for name in sampling_config_names():
        cs.store(group="sampling", name=name, node=sampling_config_factory(name))
    cs.store(name="my_run_dir", node=MyRunDir, package="hydra.run")
    cs.store(name="my_sweep_dir", node=MySweepDir, package="hydra.sweep")
    return cs


def get_mock_train_cfg(max_seq_length: int, data_dir: str) -> DictConfig:
    """
    The most typical set up of the train config file for preprocess / analysis
    """
    return OmegaConf.create(
        {
            "dataset": {
                "max_seq_length": max_seq_length,
                "data_dir": data_dir,
                "data_type": "parquet",
                "path": None,
            },
            "data": {"transforms": ["image", "shuffle"], "tokenization": False},
            "sampling": {"name": "random", "temperature": 1.0},
        }
    )
