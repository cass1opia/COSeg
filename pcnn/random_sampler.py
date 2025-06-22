__all__ = ['RandomSampler']

from .abstract_sampler import AbstractSampler
from typing import Optional
import numpy as np


class RandomSampler(AbstractSampler):
    r""" Implements a random sampling strategy.

    :param k: Number of points to be sampled from the neighborhood.
    :type k: integer
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    """
    def __init__(self, k: int, generator: Optional[np.random.Generator] = None):
        super().__init__(k, generator)

    def __call__(self, _: np.ndarray, neighborhood_indices: np.ndarray):
        r"""
        :param neighborhood_indices: The indices of the points in the given neighborhood.
                                    Must have shape :math:`(N)` where `N = number of points`
        :type neighborhood_indices: numpy.ndarray

        :return: the indices of the selected points.
        :rtype: numpy.ndarray

        """
        # If the neighborhood contains less than k elements.
        if self.k >= len(neighborhood_indices):
            return neighborhood_indices

        k_indices = self.generator.permutation(neighborhood_indices)
        k_indices = k_indices[:self.k]

        return k_indices
