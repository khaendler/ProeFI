import random

import numpy as np

from .base_reservoir_storage import BaseReservoirStorage


class UniformReservoirStorage(BaseReservoirStorage):
    """ Uniform Reservoir Storage

    Summarizes a data stream by keeping track of a fixed length reservoir of observations.
    Each past observation of the stream has an equal probability of being in the reservoir at
    the current time.
    For more information we refer to https://en.wikipedia.org/wiki/Reservoir_sampling.

    Args:
        stored_samples int: Number of samples observed in the stream.
    """

    def __init__(self, seed: int, size: int = 1000):
        super().__init__(size=size)
        self._rng = random.Random(seed)
        self.stored_samples: int = 0
        self._algo_wt = np.exp(np.log(self._rng.random()) / self.size)
        self._algo_l_counter: int = (
                self.size + (np.floor(np.log(self._rng.random()) / np.log(1 - self._algo_wt)) + 1)
        )

    def update(self, x: dict):
        """Updates the reservoir with the current sample if necessary.

        The update mechanism follows the optimal algorithm as stated here:
        https://en.wikipedia.org/wiki/Reservoir_sampling#Optimal:_Algorithm_L.

        Args:
            x (dict): Current observation's features.
        """
        self.stored_samples += 1
        if self.stored_samples <= self.size:
            self.storage.append(x)
        else:
            if self._algo_l_counter == self.stored_samples:
                self._algo_l_counter += (np.floor(
                    np.log(self._rng.random()) / np.log(1 - self._algo_wt)) + 1)
                rand_idx = self._rng.randrange(self.size)
                self.storage[rand_idx] = x
                self._algo_wt *= np.exp(np.log(self._rng.random()) / self.size)
