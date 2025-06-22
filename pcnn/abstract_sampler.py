__all__ = ['AbstractSampler']

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractSampler(ABC):
    """
    Abstract base class for PCNN's dataset subsampling strategies.

    :param k: Number of points to be sampled from the neighborhood.
    :type k: integer
    :param generator: If not `None`, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    """
    @abstractmethod
    def __init__(self, k: int, generator: Optional[np.random.Generator] = None):
        self.k = k
        if generator is None:
            generator = np.random.default_rng()
        self.generator = generator

    @abstractmethod
    def __call__(self, neighborhood: np.ndarray, neighborhood_indices: np.ndarray):
        """
        Samples a subset of points from a given neighborhood according to the implemented sampling strategy.
        Returns the indices of the selected points only.

        :param neighborhood: Array containing the neighborhood's XYZ coordinates and optionally other necessary feature
            values needed for sampling. Must have shape :math:`(N, F)`, where `F = 3 + number of optional features`,
            `N = number of points`.
        :type neighborhood: numpy.ndarray
        :param neighborhood_indices: The indices of the points in the given neighborhood.
            Must have shape :math:`(N)` where `N = number of points`.
        :type neighborhood_indices: numpy.ndarray.
        :return: the indices of the selected points. Has shape :math:`(K)`, where `K = self.k`
        :rtype: numpy.ndarray
        """
        pass
