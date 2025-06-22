""" Provides several point cloud sampling methods. """

__all__ = ['farthest_point_sample_numpy', 'farthest_point_sample_torch', 'farthest_point_sampling_cluster',
           'farthest_point_sampling_numpy', 'repeat_if_necessary',
           'stratified_sampling', 'uniform_sampling']

import math
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch

from torch_cluster import fps

NeighborhoodSampling = Literal['random_sampling', 'stratified_sampling', 'farthest_point_sampling',
                               'feature_based_sampling', 'grid_sampling', 'uniform_sampling']


def farthest_point_sample_torch(xyz: torch.Tensor,
                                k: int,
                                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    r"""
    Iterative farthest point sampling implemented in PyTorch: Samples a subset of N points
    :math:`{x_1 , x_2, ..., x_N}`, so that from the set :math:`{x_1 , x_2, ..., x_i}`, :math:`x_i` is the most distant
    point to the set of remaining points.

    :param xyz: Point coordinates from which to sample. Must have shape :math:`(B, N, 3)`, where `B = batch size`, and
                `N = number of input points`.
    :type xyz: torch.Tensor
    :param k: Number of points to be sampled.
    :type k: integer
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: torch.Generator, optional

    :return: Indices of sampled points of shape :math:`(B, num_sample)`, where `B = batch size`.
    :rtype: torch:Tensor
    """
    B, N, D = xyz.shape
    sampled_indices = torch.zeros(B, k, dtype=torch.long, device=xyz.device)
    # Create a tensor to store the minimum distance to the sampled points for each point.
    min_distance_to_sampled_points = torch.full((B, N), fill_value=torch.inf, device=xyz.device)
    # Sample the first point randomly.
    farthest_point_idx = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device, generator=generator)
    for i in range(k):
        # Store the sampled point index.
        sampled_indices[:, i] = farthest_point_idx
        # Get the point coordinates of the sampled point.
        farthest_point_xyz = torch.gather(xyz, 1, farthest_point_idx.view(B, 1, 1).expand(-1, -1, 3))
        # Compute the distance between each point and the sampled point.
        dist = torch.cdist(xyz, farthest_point_xyz).squeeze(-1)
        # For each point, update the minimum distance to one of the sampled points.
        min_distance_to_sampled_points = torch.minimum(min_distance_to_sampled_points, dist)
        # Sample the next point by choosing the point with the maximum minimum distance
        # to the sampled points.
        farthest_point_idx = torch.argmax(min_distance_to_sampled_points, -1)

    return sampled_indices


def farthest_point_sample_numpy(xyz: np.ndarray,
                                k: int,
                                generator: Optional[np.random.Generator] = None) -> np.ndarray:
    r"""
    Iterative farthest point sampling implemented in numpy: Samples a subset of N points :math:`{x_1 , x_2, ..., x_N}`,
    so that from the set :math:`{x_1 , x_2, ..., x_i}`, :math:`x_i` is the most distant point to the set of remaining
    points.

    :param xyz: Point coordinates from which to sample. Must have shape :math:`(B, N, 3)`, where `B = batch size`, and
                `N = number of input points`.
    :type xyz: numpy.ndarray
    :param k: Number of points to be sampled.
    :type k: integer
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional

    :return: Indices of sampled points of shape :math:`(B, num_sample)`, where `B = batch size`.
    :rtype: numpy.ndarray
    """
    if generator is None:
        generator = np.random.default_rng()

    B, N, D = xyz.shape
    sampled_indices = np.zeros((B, k), dtype=np.int64)
    # Create an array to store the minimum distance to the sampled points for each point.
    min_distance_to_sampled_points = np.full((B, N), fill_value=np.inf)
    # Sample the first point randomly.
    farthest_point_idx = generator.integers(0, N, (B,), dtype=np.int64)
    for i in range(k):
        # Store the sampled point index.
        sampled_indices[:, i] = farthest_point_idx
        # Get the point coordinates of the sampled point.
        farthest_point_xyz = np.take_along_axis(xyz, np.reshape(farthest_point_idx, (B, 1, 1)).repeat(3, axis=-1),
                                                axis=1)

        # Compute the distance between each point and the sampled point.
        square_dist = ((xyz - farthest_point_xyz) ** 2).sum(axis=-1)
        # For each point, update the minimum distance to one of the sampled points.
        min_distance_to_sampled_points = np.minimum(min_distance_to_sampled_points, square_dist)
        # Sample the next point by choosing the point with the maximum minimum distance
        # to the sampled points.
        farthest_point_idx = np.argmax(min_distance_to_sampled_points, axis=-1)

    return sampled_indices


def farthest_point_sampling_numpy(neighborhood_indices: np.ndarray,
                                  points: np.ndarray,
                                  k: int,
                                  generator: Optional[np.random.Generator] = None) -> np.ndarray:
    r""" Implements a farthest point sampling strategy.

    :param neighborhood_indices: The indices of the points in the given neighborhood.
      Must have shape :math:`(N)` where `N = number of points`.
    :type neighborhood_indices: numpy.ndarray
    :param points: The points in the dataset from which to sample.
                   Must have shape :math:`(N, 3)` where `N = number of points`
    :type points: numpy.ndarray
    :param k: Number of points to be sampled from the neighborhood.
    :type k: integer
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional

    :return: the indices of the selected points.
    :rtype: numpy.ndarray
    """

    # If the neighborhood contains less than k elements.
    if k >= len(neighborhood_indices):
        return neighborhood_indices

    if generator is None:
        generator = np.random.default_rng()

    # Sampling k indices from the neighborhood indices.
    k_indices = farthest_point_sample_numpy(np.expand_dims(points, axis=0), k, generator=generator).squeeze(axis=0)

    return neighborhood_indices[k_indices]


def farthest_point_sampling_cluster(neighborhood_indices: np.ndarray,
                                    points: np.ndarray,
                                    k: int,
                                    generator: Optional[np.random.Generator] = None) -> np.ndarray:
    r""" Optimized implementation of farthest point sampling in C++ / CUDA from
    https://github.com/rusty1s/pytorch_cluster/tree/master#farthestpointsampling.

    :param neighborhood_indices: The indices of the points in the given neighborhood.
      Must have shape :math:`(N)` where `N = number of points`.
    :type neighborhood_indices: numpy.ndarray
    :param points: The points in the dataset from which to sample.
        Must have shape :math:`(N, 3)` where `N = number of points`.
    :type points: numpy.ndarray
    :param k: Number of points to be sampled from the neighborhood.
    :type k: integer
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional

    :return: the indices of the selected points.
    :rtype: numpy.ndarray
    """
    # If the neighborhood contains less than k elements.
    if k >= len(neighborhood_indices):
        return neighborhood_indices

    if generator is None:
        generator = np.random.default_rng()

    # Randomly shuffling the points and indices.
    shuffle_indices = np.arange(len(neighborhood_indices))
    shuffle_indices = generator.permutation(shuffle_indices)

    # Converting the k value to a ratio.
    ratio = k / len(neighborhood_indices)
    # Creating points and batch tensors.
    points_tensor = torch.from_numpy(points[shuffle_indices])
    # Farthest point sampling call.
    index = fps(points_tensor, ratio=ratio, random_start=False)
    # Getting the sampled indices.
    sampled_indices = neighborhood_indices[shuffle_indices]
    sampled_indices = sampled_indices[index.numpy()]
    # In case of rounding errors during the ratio calculation.
    if len(sampled_indices) > k:
        sampled_indices = sampled_indices[:k]
    if len(sampled_indices) < k:
        # Obtain and add indices not yet sampled until k.
        idxs_to_add = k - len(sampled_indices)
        remaining_indices = np.setdiff1d(neighborhood_indices, sampled_indices,
                                         assume_unique=True)
        sampled_indices = np.concatenate((sampled_indices, remaining_indices[:idxs_to_add]))

    return sampled_indices


def feature_based_sampling(neighborhood_indices: np.ndarray,
                           point_feature_vals: np.ndarray,
                           k: int,
                           top_percentage: float = 0.7,
                           generator: Optional[np.random.Generator] = None,
                           return_full_array: Optional[bool] = False) -> np.ndarray:
    r""" Implements a feature-based sampling strategy. Sorts the neighborhood
    by the given feature values, and takes the specified `top_percentage` of `k` from the points
    with the highest feature values. It then fills the rest of the `k` points using random sampling.

    :param neighborhood_indices: The indices of the points in the given neighborhood.
        Must have shape :math:`(N)` where `N = number of points`.
    :type neighborhood_indices: numpy.ndarray.
    :param point_feature_vals: Array containing the feature values from which to sample. Must have
        shape :math:`(N)`, where `N = number of points`.
    :type point_feature_vals: numpy.ndarray
    :param k: Number of points to be sampled from the neighborhood.
    :type k: integer
    :param top_percentage: Percentage of k that will determine how many points with the
        highest feature values to sample. Defaults to `0.7`.
    :type top_percentage: float, optional
    :param generator: If not `None`, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    :param return_full_array: Flag indicating if we are to return the full array or clipping it
        to `k` values, in case an external method needs the full array in return. Defaults to `False`.
    :type return_full_array: bool, optional

    :return: the indices of the selected points.
    :rtype: numpy.ndarray
    """
    # Validating that `top_percentage` is in a valid range.
    assert top_percentage >= 0.0 and top_percentage <= 1.0, '`top_percentage` must be within the range [0.0, 1.0].'

    # If the neighborhood contains less than k elements.
    if k >= len(neighborhood_indices):
        return neighborhood_indices

    if generator is None:
        generator = np.random.default_rng()

    # Sorting points according to their highest `sampling_feature` value.
    sorted_points_idxs = np.flip(np.argsort(point_feature_vals))
    sorted_idxs = neighborhood_indices[sorted_points_idxs]

    # Calculating the amount of points to be sampled.
    num_top_points = math.ceil(k * top_percentage)

    # Preallocating and filling the final array.
    if return_full_array:
        k_indices = np.zeros(len(neighborhood_indices), dtype=int)
        k_indices[:num_top_points] = sorted_idxs[:num_top_points]
        k_indices[num_top_points:] = generator.permutation(sorted_idxs[num_top_points:])
    else:
        remaining_points = k - num_top_points
        k_indices = np.zeros(k, dtype=int)
        k_indices[0:num_top_points] = sorted_idxs[0:num_top_points]
        k_indices[num_top_points:k] = generator.permutation(sorted_idxs[num_top_points:])[:remaining_points]

    return k_indices


def stratified_sampling(shell_indices: List[np.ndarray],
                        k: int,
                        k_percentages: np.ndarray,
                        strategy: str,
                        top_feature_percentage: float = 0.7,
                        generator: Optional[np.random.Generator] = None,
                        shell_points: Optional[List[np.ndarray]] = None,
                        feature_idx: Optional[int] = None) -> np.ndarray:
    r"""Receives points belonging to a given number of shells and samples the specified
    percentages of `k` according to the given strategy (currently,
    concentric, point count-based and feature-based).

    :param shell_indices: List of numpy.ndarrays that contains the indices of the points
      in each shell.
    :type shell_indices: List[numpy.ndarray]
    :param k: total number of desired points in the neighborhood.
    :type k: integer
    :param k_percentages: numpy.ndarray of percentages of k to be sampled on each shell.
    :type k_percentages: numpy.ndarray
    :param strategy: Sampling strategy to be used.
    :type strategy: string
    :param top_feature_percentage: Percentage of points to sample based on their curvature values if
        `strategy` is set to `curvature`. Defaults to `0.7`.
    :type top_feature_percentage: float
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    :param shell_points: List of numpy.arrays that contains the point data for the different
        shells. Must not be `None` when strategy is set to `curvature`. Defaults to `None`.
    :type shell_points: List[numpy.ndarray], optional
    :param feature_idx: Index of the column of the feature of interest. Defaults to `None`.
    :type feature_idx: int, optional

    :return: the indices of the selected points.
    :rtype: numpy.ndarray
    """

    # In the event that we have less than k elements across all shells, we
    # just return the concatenation of the shells directly.
    total_length = sum(len(shell) for shell in shell_indices)
    if total_length < k:
        return np.concatenate(shell_indices, dtype=int)

    if generator is None:
        generator = np.random.default_rng()

    # Sanity check: we want to be sure that the number of neighborhoods (shells)
    # and of k percentages is the same (this should be ensured by previous steps
    # in the code).
    assert len(shell_indices) == len(k_percentages), 'Dimension mismatch: '\
                                                     '`shell_indices` and '\
                                                     '`k_percentages` should '\
                                                     'be of the same size.'

    # Sorting the k_percentages in decreasing order. We need to invert the
    # array so that the `sort()` method returns the array sorted in
    # decreasing order rather than increasing.
    k_percentages[::-1].sort()
    # Getting the actual k values per shell.
    k_values = []
    # The last one (the most sparse (small) k value) will be handled
    # differently to ensure that the sum of all values is equal to k.
    for i in range(len(k_percentages) - 1):
        k_values.append(math.ceil(k * (k_percentages[i] / 100.0)))
    k_values.append(k - sum(k_values))

    if "point_count" in strategy:
        # We assign the percentages of k according to the number of points in
        # each shell instead.
        shell_indices.sort(key=len, reverse=True)
    elif "concentric" in strategy:
        # We don't need to do anything in this case.
        pass
    else:
        # Validating that the points and column names are passed for feature-based
        # sampling.
        assert shell_points is not None and feature_idx is not None, 'The shell_points '\
                                                                     'point_features are needed '\
                                                                     'for feature-based '\
                                                                     'sampling.'

        # Calculating mean feature value per shell.
        mean_feature_values = [np.mean(points, axis=0)[feature_idx] for points in shell_points]
        # Obtaining the indexes of the sorted mean feature values in decreasing order.
        sorted_feature_idxs = np.flip(np.argsort(mean_feature_values)).astype(int)
        # Sorting the shell indices and points according to the mean feature values.
        shell_indices = np.asarray(shell_indices, dtype=object)[sorted_feature_idxs]
        shell_points = np.asarray(shell_points, dtype=object)[sorted_feature_idxs]

    # Preallocating the final array.
    k_indices = np.zeros(k, dtype=int)
    lower_limit = 0
    true_len = 0
    # Sampling the denser regions with the bigger k values.
    for idx, k_val in enumerate(k_values):
        upper_limit = len(shell_indices[idx][:k_val]) + lower_limit
        if "concentric" in strategy or "point_count" in strategy:
            # We shuffle in-place, as we want to keep this random ordering for the
            # next step.
            generator.shuffle(shell_indices[idx])
        else:
            # We call the `feature_based_sampling` method for each shell. We want to keep
            # the new order of the shell indices for the next step.
            shell_indices[idx] = feature_based_sampling(shell_indices[idx],
                                                        shell_points[idx][:, feature_idx],  # type: ignore[index]
                                                        k_val,
                                                        top_feature_percentage,
                                                        generator,
                                                        return_full_array=True)
        k_indices[lower_limit:upper_limit] = shell_indices[idx][:k_val]
        lower_limit = upper_limit
        true_len += len(shell_indices[idx][:k_val])

    # If after the sampling procedure we ended up with less than k elements, we want
    # to return as close to k elements as we can by using unsampled points from the
    # different shells.
    if true_len < k:
        lower_limit = true_len
        for idx, k_val in enumerate(k_values):
            upper_limit = len(shell_indices[idx][k_val:]) + lower_limit
            # Checking if in the next move we will fill k_indices up to k elements.
            if upper_limit > k:
                # If the amount of points left in the current shell will put us above
                # k elements, we only take as much as we need and exit.
                cut_index = k_val + k - lower_limit
                k_indices[lower_limit:] = shell_indices[idx][k_val:cut_index]
                true_len = k
                break
            elif len(shell_indices[idx][k_val:]) == 0:
                # If there are no unsampled elements on this shell, nothing to be done.
                pass
            else:
                # If elements still unsampled on this shell and not yet filled the
                # required k_values, we add all the elements of the current shell
                # and continue.
                k_indices[lower_limit:upper_limit] = shell_indices[idx][k_val:]
                lower_limit = upper_limit
                true_len += len(shell_indices[idx][k_val:])

    # Clipping to the actual amount of indices sampled in case all shells together contain
    # less than k elements.
    if true_len < k:
        k_indices = k_indices[:true_len]

    return k_indices


def uniform_sampling(points: np.ndarray,
                     indices: np.ndarray,
                     num_points: int,
                     eps: float = 1e-4,
                     generator: Optional[np.random.Generator] = None) -> np.ndarray:
    r"""
    Uniformly samples :attr:`num_points` points from the input points. The sampling is done by dividing the space into
    approximately :attr:`num_points` buckets of equal size and randomly selecting one point from each
    filled bucket. If this results in less than `num_points`, additional points are drawn randomly from the filled
    buckets.

    To approximate the size and number of buckets that are needed to cover the bounding box volume of the point cloud,
    the following approach is used: Let :math:`w_1`, :math:`w_2`, and :math:`w_3` be the width, height, and depth of
    the point cloud's bounding box. Then, the aspect ratios of the bounding box relative to :math:`w_1` can be defined:
    :math:`r_2 = \frac{w_2}{w_1}` and :math:`r_3 = \frac{w_3}{w_1}`. Next, we assume that the bounding box of the point
    cloud volume can be filled exactly by :math:`k` cubes with edge length :math:`x`, where :math:`k` is set to the
    value of :attr:`num_points`:

    :math:`k \cdot x^3 = w_1 \cdot w_2 \cdot w_3 = w_1^3 \cdot r_2 \cdot r_3`

    Assuming that the volume of the bounding box is exactly filled by the cubes, :math:`w_1` is a multiple of :math:`x`,
    i.e., :math:`w_1 = x \cdot n_1`:

    :math:`k \cdot \frac{w_1^3}{n_1^3} = w_1^3 \cdot r_2 \cdot r_3`

    This gives us the number of cubes along each dimension of the bounding box:

    :math:`n_1 = \sqrt[3]{\frac{k}{r_2 \cdot r_3}}`

    :math:`n_2 = n_1 \cdot r_2`

    :math:`n_3 = n_1 \cdot r_3`

    where the edge length of the cubes is

    :math:`x = \frac{w_1}{n_1}`

    The assumption that the volume of the bounding box can be exactly filled by :math:`k` equal-sized cubes holds only
    if :math:`w_1`, :math:`w_2` and :math:`w_3` are multiples of :math:`x`, i.e., :math:`n_1`, :math:`n_2`, and
    :math:`n_3` are integers. If this is not the case, we ensure that the cubes are sufficiently large to cover the
    entire bounding box by flooring the numbers.

    :param points: The points in the dataset from which to sample.
        Must have shape :math:`(N, 3)` where `N = number of points`.
    :type points: numpy.ndarray
    :param indices: The indices of the points in the given neighborhood.
        Must have shape :math:`(N)` where `N = number of points`.
    :type indices: numpy.ndarray
    :param num_points: Number of points to sample.
    :type num_points: int
    :param eps: Small offset to be added to the sampling volume to avoid that new buckets are started exactly at the
        border of the sampling volume.
    :type eps: float, optional
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :return: Indices of the sampled points of shape `(num_points)`.
    :rtype: numpy.ndarray
    """
    # If the neighborhood contains less than k elements.
    if num_points <= 0:
        return np.empty((0, ), dtype=indices.dtype)
    elif num_points >= len(indices):
        return indices
    if generator is None:
        generator = np.random.default_rng()

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Compute the range of each dimension
    ranges = max_coords - min_coords + eps

    # Compute aspect ratios of bounding box
    aspect_ratios = (ranges / ranges[0])

    # Compute how many grid cells along each dimension are needed to cover the bounding box
    num_cells = (num_points / (aspect_ratios[1] * aspect_ratios[2])) ** (1 / 3) * aspect_ratios
    num_cells = np.floor(num_cells)

    # Compute size of grid cells that is needed to cover at least the whole sampling volume
    stride = np.max(ranges / num_cells)

    # Shuffle the indices and coordinates
    shuffled_indices = generator.permutation(len(points))
    points = points[shuffled_indices]
    indices = indices[shuffled_indices]

    # Compute the bucket index for each point
    bucket_index = np.floor((points - min_coords) / stride).astype(int)

    # Randomly draw one point from each bucket
    _, selected_indices = np.unique(bucket_index, axis=0, return_index=True)
    selected_indices = indices[selected_indices]

    # If didn't select enough points, add more from the non-selected indices
    if len(selected_indices) < num_points:
        non_selected_indices = np.setdiff1d(indices, selected_indices)
        selected_indices = np.concatenate((selected_indices, non_selected_indices[:num_points - len(selected_indices)]))

    return selected_indices


def repeat_if_necessary(indices: np.ndarray,
                        k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Repeats points if necessary up to a given `k` value.

    :param indices: the indices of the points in the given neighborhood.
                    Must have shape :math:`(N)` where `N = number of points`
    :type indices: numpy.ndarray
    :param k: number of desired points in the neighborhood.
    :type k: integer

    :return: an array with the repeated data and a boolean mask indicating which indices are duplicates.
      Both arrays have length `k`.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    data = indices
    repeated_data = np.tile(data, np.ceil(k / len(data)).astype(int))
    is_duplicate = np.zeros(len(repeated_data), dtype=bool)
    is_duplicate[len(data):] = True
    # Clipping to k elements.
    repeated_data = repeated_data[:k]
    is_duplicate = is_duplicate[:k]

    return repeated_data, is_duplicate
