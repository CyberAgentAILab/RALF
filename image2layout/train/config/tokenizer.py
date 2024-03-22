from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    num_bin: int = 128
    var_order: tuple[str, str, str, str, str] = (
        "label",
        "width",
        "height",
        "center_x",
        "center_y",
    )
    pad_until_max: bool = (
        False  # True for diffusion models, False for others for efficient batching
    )
    special_tokens: list[str] = field(default_factory=lambda: ["pad", "bos", "eos"])
    is_loc_vocab_shared: bool = False
    geo_quantization: str = "linear"
