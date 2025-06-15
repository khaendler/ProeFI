from typing import List
from abc import abstractmethod, ABC


class BaseReservoirStorage(ABC):
    """ Reservoir Storage - base class

    Parts of this code are adapted from https://github.com/mmschlk/iXAI.

    Args:
        size (int): Size of the reservoir.

    """
    def __init__(self, size: int):
        self.storage: List[dict] = []
        self.size = size

    @abstractmethod
    def update(self, x: dict):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        return self.storage[idx]

    def __len__(self):
        return len(self.storage)

