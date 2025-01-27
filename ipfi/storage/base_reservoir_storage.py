from typing import List
from abc import abstractmethod, ABC


class BaseReservoirStorage(ABC):

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

