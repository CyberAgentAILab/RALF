from dataclasses import dataclass
from typing import List


@dataclass
class BaseSamplingConfig:
    name: str = ""


@dataclass
class DeterministicSamplingConfig(BaseSamplingConfig):
    name: str = "deterministic"


@dataclass
class StochasticSamplingConfig(BaseSamplingConfig):
    temperature: float = 1.0


@dataclass
class RandomSamplingConfig(StochasticSamplingConfig):
    name: str = "random"
    num_timesteps: int = 50


@dataclass
class GumbelSamplingConfig(StochasticSamplingConfig):
    name: str = "gumbel"


@dataclass
class TopKSamplingConfig(StochasticSamplingConfig):
    name: str = "top_k"
    top_k: int = 5
    temperature: float = 1.0


@dataclass
class TopPSamplingConfig(StochasticSamplingConfig):
    name: str = "top_p"
    top_p: float = 0.9


@dataclass
class TopKTopPSamplingConfig(StochasticSamplingConfig):
    name: str = "top_k_top_p"
    top_k: int = 5
    top_p: float = 0.9


_SAMPLING_CONFIG = {
    "top_k": TopKSamplingConfig,
    "top_k_top_p": TopKTopPSamplingConfig,
    "top_p": TopPSamplingConfig,
    "deterministic": DeterministicSamplingConfig,
    "random": RandomSamplingConfig,
    "gumbel": GumbelSamplingConfig,
}


def sampling_config_factory(name: str) -> BaseSamplingConfig:
    return _SAMPLING_CONFIG[name]


def sampling_config_names() -> List[str]:
    return list(_SAMPLING_CONFIG.keys())


if __name__ == "__main__":
    import torch
    from einops import repeat
    from image2layout.train.helpers.sampling import sample
    from omegaconf import OmegaConf

    sampling_cfg = OmegaConf.create({"name": "top_p", "top_p": 0.9, "temperature": 1.0})
    logits = repeat(torch.arange(5), "c -> b c 1", b=2)
    x = sample(logits, sampling_cfg, return_confidence=True)
    print(x)
