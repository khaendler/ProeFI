import random

from .base_reservoir_storage import BaseReservoirStorage


class GeometricReservoirStorage(BaseReservoirStorage):
    """ Geometric Reservoir Storage

    Parts of this code are adapted from https://github.com/mmschlk/iXAI.

    Args:
        size (int): Size of the reservoir.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, size: int, seed: int):
        super().__init__(size=size)
        self._rng = random.Random(seed)
        self._constant_probability = 1 / self.size

    def update(self, x: dict):
        if len(self.storage) < self.size:
            self.storage.append(x)
        else:
            random_float = self._rng.random()
            if random_float <= self._constant_probability:
                rand_idx = self._rng.randrange(self.size)
                self.storage[rand_idx] = x
