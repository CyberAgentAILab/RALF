from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


class VoidScheduler(LambdaLR):
    """
    A scheduler that does nothing (by multiplying 1.0 in every epoch).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,  # to ignore some args that are not neccessary
    ) -> None:
        super().__init__(optimizer=optimizer, lr_lambda=lambda epoch: 1.0)
