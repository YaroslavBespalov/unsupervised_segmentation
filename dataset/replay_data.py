from typing import Tuple, List, Generator

import torch
from torch import nn, Tensor


class ReplayBuffer:

    def __init__(self, tuple_size: int):
        self.data: List[Tuple[Tensor]] = []
        self.tuple_size = tuple_size

    def append(self, *obs: Tensor):
        assert obs.__len__() == self.tuple_size
        self.data.append(obs)

    def size(self) -> int:
        return self.data.__len__()

    def sample(self, n: int):

        return (
            torch.cat(
                [d[k] for d in self.data[self.size() - n:]]
            ).cuda() for k in range(self.tuple_size)
        )
