import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DSGANScheduler(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        gamma: float = 0.8,
        network: str = "generator",
        **kwargs: Any,  # to ignore some args that are not neccessary
    ):
        # A maximum epoch and a learning rate is unchanged bewteen CGL and PKU dataset.
        # hardcoded for now
        assert epochs == 300
        if network == "generator":
            milestones = torch.arange(0, epochs, 50)
        elif network == "discriminator":
            milestones = torch.arange(0, epochs, 25)
        else:
            raise ValueError(f"Unknown network: {network}")

        fname = "torch.optim.lr_scheduler.MultiStepLR"
        logger.info(
            f"Launch {fname} for {network} with {epochs=} {milestones=} {gamma=}"
        )
        super().__init__(optimizer=optimizer, milestones=milestones, gamma=gamma)
