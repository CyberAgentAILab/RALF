# strictly follow the following link
# https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
import os
import random

import torch
from torch.distributed import init_process_group


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(9900, 9999))
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class DDPWrapper(torch.nn.parallel.DistributedDataParallel):
    """
    Very shallow wrapper to access attributes of a model
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
