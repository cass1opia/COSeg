__all__ = ['PointCloud']

import logging
import sys
from typing import Optional

import pandas as pd
import numpy as np


class PointCloud:
    """Data representation of point clouds with data and shift. This is the used representation for everything up until
    :code:`PointCloudDataset`. Point clouds are required to have the columns :code:`[x, y, z]` in that order to be
    initialized as a :code:`PointCloud` object.

    :param dataset: point cloud data
    :type dataset: pandas.DataFrame, optional
    :param x_shift: x shift of point cloud, defaults to 0
    :type x_shift: np.float, optional
    :param y_shift: y shift of point cloud, defaults to 0
    :type y_shift: np.float, optional
    :param z_shift: z shift of point cloud, defaults to 0
    :type z_shift: np.float, optional
    :param x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter.
    :type x_max_resolution: float, optional
    :param y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter.
    :type y_max_resolution: float, optional
    :param z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter.
    :type z_max_resolution: float, optional
    """

    min_supported_columns = ['x', 'y', 'z']

    def __init__(self,
                 dataset: Optional[pd.DataFrame] = None,
                 x_shift: Optional[int] = None,
                 y_shift: Optional[int] = None,
                 z_shift: Optional[int] = None,
                 x_max_resolution: Optional[float] = None,
                 y_max_resolution: Optional[float] = None,
                 z_max_resolution: Optional[float] = None) -> None:

        self._dataset = dataset if dataset is not None else pd.DataFrame(columns=PointCloud.min_supported_columns)
        self.x_shift = x_shift if x_shift else 0
        self.y_shift = y_shift if y_shift else 0
        self.z_shift = z_shift if z_shift else 0
        self.x_max_resolution = x_max_resolution
        self.y_max_resolution = y_max_resolution
        self.z_max_resolution = z_max_resolution
        self.update_shift()
        if not self.valid():
            error_code = "E703"
            logging.error("(Error {}) PointCloud object was initialized with a wrongly formatted dataset object."
                          .format(error_code))
            sys.exit(error_code)

    @property
    def _dataset(self) -> pd.DataFrame:
        return self._dataset_df

    @_dataset.setter
    def _dataset(self, dataset: pd.DataFrame) -> None:
        self._dataset_df = dataset
        self._dataset_df.columns = self._dataset_df.columns.astype(str)

    def update_data(self, dataset: pd.DataFrame) -> None:
        self._dataset = dataset
        if not self.valid():
            error_code = "E704"
            logging.error("(Error {}) PointCloud object ended up not having the right format after updating data."
                          .format(error_code))
            sys.exit(error_code)

    def update_shift(self, x_shift: None = None, y_shift: None = None, z_shift: None = None) -> None:
        """Updates internal shift with new shift variables.

        :param x_shift: x shift of point cloud, defaults to no change
        :type x_shift: np.float, optional
        :param y_shift: y shift of point cloud, defaults to no change
        :type y_shift: np.float, optional
        :param z_shift: z shift of point cloud, defaults to no change
        :type z_shift: np.float, optional
        """
        if x_shift is not None:
            self.x_shift = x_shift
        if y_shift is not None:
            self.y_shift = y_shift
        if z_shift is not None:
            self.z_shift = z_shift
        self._update_shift_vector()

    def _update_shift_vector(self) -> None:
        """Updates shift vector with actual shift variables and ensures length matches dataset column length."""
        self._shift_vector = ([self.x_shift, self.y_shift, self.z_shift] + [0] * (len(self._dataset.columns) - 3))

    def valid(self) -> bool:
        """Checks point cloud data for valid column order.

        :return: returns `True` if column order is valid, `False` otherwise
        :rtype: bool
        """
        # just in case our dataset is None
        if self._dataset is None:
            return False
        # need at the very least 3 columns (x, y, z), so catch every case with fewer
        if len(self._dataset.columns) < 3:
            return False
        if not all(isinstance(column, str) for column in self._dataset.columns):
            return False
        #  throws an error if there are no columns on older pandas versions (tested on 0.24.1)
        cols = [column.lower() for column in self._dataset.columns]
        if not np.array_equal(PointCloud.min_supported_columns[:3], cols[:3]):
            return False
        # For more than xyz as min_support
        for column in PointCloud.min_supported_columns[3:]:
            if column not in cols:
                return False
        return True

    def data(self) -> pd.DataFrame:
        """ This method returns the dataset object. Changing the dataset object also changes the point cloud's state.
        Use with caution when using it without :code:`update_data`.

        :return: point cloud data
        :rtype: pandas.DataFrame
        """
        return self._dataset

    def data_shifted(self) -> pd.DataFrame:
        """
        :return: point cloud data with shift applied
        :rtype: pandas.DataFrame
        """
        if len(self._dataset) > 0:
            if len(self._shift_vector) != len(self._dataset.columns):
                self._update_shift_vector()
            return (self._dataset + self._shift_vector).copy()
        return self._dataset

    def values_shifted(self) -> np.ndarray:
        """
        :return: point cloud data with shift applied, as values
        :rtype: np.array
        """
        return self.data_shifted().values

    def __len__(self) -> int:
        return len(self._dataset)

    def __eq__(self, other: object) -> bool:
        """Checks for equality of shifted point cloud prepresentation.

        :return: `False` if shifted point cloud data is not equal, `True` otherwise
        :rtype: bool
        """
        if not isinstance(other, PointCloud):
            return False
        return self.data_shifted().equals(other.data_shifted()) and \
               all(self.data().dtypes == other.data().dtypes)

    def almost_equal(self, other: object, rounding: int = 4) -> bool:
        """Checks for equality of shifted point cloud prepresentation.

        :return: `False` if shifted point cloud data is not equal, `True` otherwise
        :rtype: bool
        """
        if not isinstance(other, PointCloud):
            return False
        return self.data_shifted().round(rounding).equals(other.data_shifted().round(rounding)) and \
               all(self.data().dtypes == other.data().dtypes)
