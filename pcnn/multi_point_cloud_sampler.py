""" Utilities for calculating weights for sampling centroids from multiple point clouds. """
__all__ = ['MultiPointCloudSampler', 'SamplingWeight']

import logging
from typing import Dict, Literal, Optional

import iteround
import numpy as np

SamplingWeight = Literal['unweighted', 'square_root', 'balanced']


class MultiPointCloudSampler:
    """
    Calculates how many samples / batches should be drawn from each point cloud.

    :param class_counts_per_file: Number of points per file and class. Must have shape `(N, C)`, where
        `N = number of files` and `C = number of classes`.
    :type class_counts_per_file: numpy.ndarray
    :param num_samples: Number of points to be drawn in total. Must be greater than zero. If the `batch_size` is not
        `None`, the number of samples to be drawn should be a multiple of the specified batch size. If this is not the
        case, the number of samples to be drawn is rounded up to the next larger multiple of the batch size.
    :type num_samples: integer
    :param weight: How different classes should be weighted when drawing samples from the point clouds: `'unweighted'` |
        `'square_root'` | `'balanced'`.

        - `'unweighted'`: The points are drawn randomly without considering class labels.
        - `'square_root'`: The number of samples for each class is chosen so that the class frequencies of the sampled
          points are approximately proportional to the square root of the class frequencies before sampling.
        - `'balanced'`: Approximately the same number of points is drawn from each class, i.e.,
            `num_samples / number of classes`.
    :type weight: string
    :param batch_size: Batch size. If the batch size is specified and the number of samples to be drawn is not a
        multiple of the batch size, the number of samples is rounded up to the next larger multiple of the batch size.
        If the batch size is specified, the number of samples to be drawn from each point cloud is rounded to a
        multiple of the batch size and it is ensured that at least one batch is drawn from each point cloud.
    :type batch_size: integer, optional
    """

    def __init__(self,
                 class_counts_per_file: np.ndarray,
                 num_samples: int,
                 weight: SamplingWeight,
                 batch_size: Optional[int] = None):
        if weight not in ['unweighted', 'square_root', 'balanced']:
            raise ValueError(f'Invalid value for weight: {weight}.')
        if num_samples <= 0:
            raise ValueError('The number of samples to be drawn from the datasets must be greater than zero.')
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError('Batch size must be a positive integer.')
            if num_samples % batch_size > 0:
                logging.warning("The number of samples to be drawn from the datasets is not a multiple of the batch "
                                "size. Rounding up to the next larger multiple of the batch size.")
                num_samples = np.ceil(num_samples / batch_size) * batch_size
        for idx in range(len(class_counts_per_file)):
            if class_counts_per_file[idx].sum() <= 0:
                raise ValueError(f'File with index {idx} does not contain any valid labels.')

        self._batch_size = batch_size
        self._class_counts_per_file = class_counts_per_file
        self._num_samples = num_samples
        self._weight = weight

        self._num_samples_per_file_and_class = self.calculate_num_samples_per_file_and_class(class_counts_per_file,
                                                                                             num_samples, weight)
        self._sampling_weights = self.calculate_sampling_weights_per_file(self._class_counts_per_file,
                                                                          self._num_samples_per_file_and_class)

    def num_samples_per_file(self, file_idx: int) -> int:
        """
        Returns number of points to be drawn from a given file.

        :param file_idx: Index of the file for which the the number of samples should be returned.
        :type file_idx: integer
        :return: Number of points to be drawn from the given file.
        :rtype: integer
        """

        num_samples_per_file = self._num_samples_per_file_and_class.sum(axis=-1)
        if self._batch_size is None:
            # round number of samples so that its sum is equal to the total number of samples to be drawn
            rounded_num_samples_per_file = iteround.saferound(num_samples_per_file, 0)
            return int(rounded_num_samples_per_file[file_idx])

        # round number of samples to be drawn from each file to a multiple of the batch size
        num_batches_per_file = num_samples_per_file / self._batch_size
        num_batches_per_file_sum = num_batches_per_file.sum()
        # ensure that at least one full batch is drawn from each point cloud
        num_batches_per_file[num_batches_per_file < 1] = 1
        num_batches_per_file = iteround.saferound(num_batches_per_file, 0, topline=num_batches_per_file_sum)
        return int(self._batch_size * num_batches_per_file[file_idx])

    def total_samples(self) -> int:
        """
        Returns total number of points to be drawn from all files.

        :return Total number of points to be drawn.
        :rtype: integer
        """

        total_samples = 0

        for file_idx in range(len(self._num_samples_per_file_and_class)):
            total_samples += self.num_samples_per_file(file_idx)

        return total_samples

    def total_batches(self) -> int:
        """
        Returns total number of batches to be drawn from all files.

        :return: Total number of batches to draw.
        :rtype: integer
        """

        assert self._batch_size is not None, 'The total number of batches can only be calculated when the batch size ' \
                                             'is specified.'

        total_batches = 0

        for file_idx in range(len(self._num_samples_per_file_and_class)):
            samples_from_file = self.num_samples_per_file(file_idx)
            assert samples_from_file % self._batch_size == 0, f'The number of samples to be drawn from file with ' \
                                                              f'index {file_idx} is not a multiple of the batch size.'
            total_batches += int(samples_from_file / self._batch_size)

        return total_batches

    def per_class_sampling_weights(self, file_idx: int) -> np.ndarray:
        """
        Returns per-class sampling weights for each class for a given file.

        :param file_idx: Index of the file for which the sampling weights should be returned.
        :type file_idx: integer
        :return: Sampling weights for the given file of shape `(C)`, where `C = number of classes`.
        :rtype: numpy.ndarray
        """
        return self._num_samples_per_file_and_class[file_idx]

    def per_point_sampling_weights(self, file_idx: int) -> Dict[int, float]:
        """
        Returns per-point sampling weights for each class for a given file. The per-point sampling weights are obtained
        by dividing the per-class sampling weights by the number of points in each class.

        :param file_idx: Index of the file for which the sampling weights should be returned.
        :type file_idx: integer
        :return: Dictionary mapping class indices to sampling weights for the given file. The dictionary contains `C`
            entries, where `C = number of classes`. The weights are per-point sampling weights (each point belonging to
            the same class is assigned the same sampling weight).
        :rtype: Dict[int, float]
        """
        return {class_idx: weight for class_idx, weight in enumerate(self._sampling_weights[file_idx])}

    @staticmethod
    def calculate_num_samples_per_file_and_class(class_counts_per_file,
                                                 num_samples: int,
                                                 weight: Literal['unweighted', 'balanced', 'square_root']
                                                 ) -> np.ndarray:
        """
        Calculates number of points to be drawn from each point cloud file for each class.

        :param class_counts_per_file: Number of points per class and file. Must have shape `(N, C)`, where
            `N = number of files`, and `C = number of classes`.
        :type class_counts_per_file: numpy.ndarray
        :param num_samples: Number of points to be drawn in total.
        :type num_samples: integer
        :param weight: How different classes should be weighted when drawing samples from the point clouds:
            `'unweighted'` | `'square_root'` | `'balanced'`.

            - `'unweighted'`: The points are drawn randomly without considering class labels.
            - `'square_root'`: The number of samples for each class is chosen so that the class frequencies of the
              sampled points are approximately proportional to the square root of the class frequencies before
              sampling.
            - `'balanced'`: Approximately the same number of points is drawn from each class, i.e.,
              `num_samples / number of classes`.
        :type weight: string
        :return: Number of points to be drawn from each point cloud file for each class. Has shape `(N, C)`, where
            `N = number of files`, and `C = number of classes`.
        :rtype: numpy.ndarray
        """

        total_class_counts = class_counts_per_file.sum(axis=0)
        total_points = total_class_counts.sum()

        if weight == 'unweighted':
            num_points_per_file_and_class = num_samples * class_counts_per_file / total_points
            return num_points_per_file_and_class

        class_frequencies = total_class_counts / total_points
        if weight == 'square_root':
            target_class_frequencies = np.sqrt(class_frequencies)
        else:
            target_class_frequencies = np.ones_like(class_frequencies)

        # normalize values
        target_class_frequencies = target_class_frequencies / target_class_frequencies.sum()

        num_points_per_class = num_samples * target_class_frequencies

        total_class_counts_divisor = np.expand_dims(np.copy(total_class_counts), 0)
        total_class_counts_divisor[total_class_counts_divisor == 0] = 1  # avoid division by zero
        num_points_per_file_and_class = class_counts_per_file / total_class_counts_divisor

        num_points_per_file_and_class *= np.expand_dims(num_points_per_class, 0)

        return num_points_per_file_and_class

    @staticmethod
    def calculate_sampling_weights_per_file(class_counts_per_file: np.ndarray,
                                            num_samples_per_file_and_class: np.ndarray) -> np.ndarray:
        """
        Calculates sampling weights for each file and class.

        :param class_counts_per_file: Number of points per file and class. Must have shape `(N, C)`, where
            `N = number of files` and `C = number of classes`.
        :type class_counts_per_file: numpy.ndarray
        :param num_samples_per_file_and_class: Number of points to be drawn from each file and class. Must have shape
            `(N, C)`, where `N = number of files`, and `C = number of classes`.
        :type num_samples_per_file_and_class: numpy.ndarray
        :return: Sampling weight for each file and class. Has shape `(N, C)`, where `N = number of files`, and
            `C = number of classes`. The weights are per-point sampling weights (each point belonging to
            the same class is assigned the same sampling weight).
        """

        class_counts_divisor = np.copy(class_counts_per_file)
        class_counts_divisor[class_counts_divisor == 0] = 1  # avoid division by zero
        sampling_weights = num_samples_per_file_and_class / class_counts_divisor

        sampling_weights_normalized = sampling_weights / sampling_weights.sum(axis=-1, keepdims=True)
        sampling_weights_normalized[sampling_weights.sum(axis=-1) == 0] = 0

        return sampling_weights_normalized
