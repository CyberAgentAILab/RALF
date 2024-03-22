from dataclasses import dataclass, field


@dataclass
class DataConfig:
    transforms: list[str] = field(default_factory=lambda: [])
    tokenization: bool = False
