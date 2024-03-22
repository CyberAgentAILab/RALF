import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ReduceLROnPlateauScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(  # type: ignore
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 2,
        threshold: float = 1e-2,
        **kwargs: Any,  # to ignore some args that are not neccessary
    ) -> None:
        fname = "torch.optim.lr_scheduler.ReduceLROnPlateau"
        logger.info(f"Launch {fname} with {mode=} {factor=} {patience=} {threshold=}")
        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
        )

    def get_last_lr(self) -> list[float]:
        # have no idea why, but there isn't
        # https://discuss.pytorch.org/t/shouldnt-reducelronplateau-super-optimizer-in-its-init/89390
        return self._last_lr  # type: ignore
