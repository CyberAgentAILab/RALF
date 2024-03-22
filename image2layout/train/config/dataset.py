from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class BaseDatasetConfig:
    max_seq_length: int = 10

    # for Huggingface Hub
    path: Optional[str] = ""

    # for local dataset
    data_dir: Optional[str] = ""
    data_type: Optional[str] = "parquet"


@dataclass
class PKUConfig(BaseDatasetConfig):
    name: str = "pku"


@dataclass
class CGLConfig(BaseDatasetConfig):
    name: str = "cgl"


# two configs with varying val/test split size might be implemented in the future
# "pku10_mid": PKUConfig,
# "pku10_large": PKUConfig,
_DATASET_CONFIG = {
    "pku10": PKUConfig,
    "pkuold10": PKUConfig,
    "cgl": CGLConfig,
}


def dataset_config_factory(name: str) -> Any:
    return _DATASET_CONFIG[name]


def dataset_config_names() -> List[str]:
    return list(_DATASET_CONFIG.keys())
