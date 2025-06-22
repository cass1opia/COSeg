from .abstract_sampler import AbstractSampler
from typing import Optional
import numpy as np

from .preprocessing_utils import grid_subsampling_np


class GridSampler(AbstractSampler):
    """
    Samples a fixed number of points from a point cloud using a combination of grid subsampling and random sampling:
    First, the point cloud is downsampled with the specified minimum grid size (one point per grid cell is randomly
    selected).  The grid size is doubled until the number of points sampled is less than or equal to the number of
    points to be sampled or the number of doubling steps is `num_grid_size_levels`. If the number of points
    sampled by the grid subsampling is less than the number of points to
    be sampled, additional points are selected randomly. If the number of points sampled by the grid subsampling
    is larger than the number of points to be sampled, points are removed randomly.

    :param k: Number of points to be sampled from the neighborhood.
    :type k: integer
    :param min_grid_size: Initial grid size for downsampling in meter.
    :type min_grid_size: float
    :param num_grid_size_levels: Specifies the number of grid sizes to test before applying random sampling. For the
        first grid subsampling step, `min_grid_size` is used. For each subsequent step, the grid size is doubled. If set
        to `None`, the grid size is doubled until the number of sampled points is smaller or equal to `k`. Defaults to
        10.
    :type num_grid_size_levels: int, optional
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    """
    def __init__(self,
                 k: int,
                 min_grid_size: float,
                 num_grid_size_levels: int = 10,
                 generator: Optional[np.random.Generator] = None):
        super().__init__(k, generator)
        self.min_grid_size = min_grid_size
        self.num_grid_size_levels = num_grid_size_levels

    def __call__(self, neighborhood: np.ndarray, neighborhood_indices: np.ndarray):
        """
        :param neighborhood_indices: The indices of the points in the given neighborhood.
            Must have shape :math:`(N)` where `N = number of points`.
        :type neighborhood_indices: numpy.ndarray
        :param points: The points in the dataset from which to sample.
            Must have shape :math:`(N, 3)` where `N = number of points`.
        :type points: numpy.ndarray

        :return: the indices of the selected points.
        :rtype: numpy.ndarray
        """
        # If the neighborhood contains less than k elements.
        if self.k >= len(neighborhood_indices):
            return neighborhood_indices

        shuffled_indices = self.generator.permutation(np.arange(0, len(neighborhood_indices)))
        sampled_indices = neighborhood_indices[shuffled_indices]
        points = neighborhood[shuffled_indices]

        i = 0
        while i < self.num_grid_size_levels:
            grid_size = self.min_grid_size * (2 ** i)
            i += 1

            points, indices = grid_subsampling_np(points, grid_size, fast_density=True,
                                                  return_sorted=False)

            sampled_indices = sampled_indices[indices]

            if len(sampled_indices) <= self.k:
                break

        if len(sampled_indices) < self.k:
            idxs_to_add = self.k - len(sampled_indices)
            remaining_indices = np.setdiff1d(neighborhood_indices, sampled_indices,
                                             assume_unique=True)
            remaining_indices = self.generator.permutation(remaining_indices)
            sampled_indices = np.concatenate((sampled_indices, remaining_indices[:idxs_to_add]))
        else:
            sampled_indices = sampled_indices[:self.k]

        return sampled_indices
