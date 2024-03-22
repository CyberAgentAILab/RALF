import logging
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)


class MultiStepLRScheduler(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        milestones: Optional[Union[list[float], list[int]]] = None,
        milestone_interval: Optional[float] = None,
        gamma: float = 0.1,
        **kwargs: Any,  # to ignore some args that are not neccessary
    ) -> None:
        assert (milestones is not None) ^ (milestone_interval is not None)
        if milestones is not None:
            if isinstance(milestones[0], float):
                # a milestone indicate the percentage of the total epochs
                assert all(0.0 <= m <= 1.0 for m in milestones)
                _milestones = [int(m * epochs) for m in milestones]
            elif isinstance(milestones[0], int):
                assert all(0.0 <= m for m in milestones)
                # a milestone indicate the absolute epoch number
                _milestones = [int(m) for m in milestones]
            else:
                raise NotImplementedError
        elif milestone_interval is not None:
            raise NotImplementedError
            # assert isinstance(milestone_interval, float)
            # assert 0.0 < milestone_interval < 1.0
            # milestones = []  # type: ignore
            # i = 0
            # while True:
            #     i += 1
            #     if (ratio := i * milestone_interval) >= 1.0:
            #         break
            #     else:
            #         milestones.append(int(ratio * epochs))  # type: ignore
        fname = "torch.optim.lr_scheduler.MultiStepLR"
        logger.info(f"Launch {fname} with {epochs=} milestones={_milestones} {gamma=}")
        super().__init__(optimizer=optimizer, milestones=_milestones, gamma=gamma)
