__all__ = ['get_prediction_dataloader', 'PointCloudDataset', 'PredictionPointCloudDataset']

from collections import deque
import hashlib
import logging
from pathlib import Path
import pickle
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import torch
from torch.multiprocessing import Lock
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate

from .data_utils import jitter, random_scale, random_rotate, shift, translate_scale_xy, translate_scale_xyz
from . import general_utils as utils
from .file_handler import FileReaderManager
from .sampling_utils import repeat_if_necessary, farthest_point_sampling_cluster, \
    stratified_sampling, NeighborhoodSampling, feature_based_sampling, uniform_sampling

from .random_sampler import RandomSampler
from .grid_sampler import GridSampler
from .abstract_sampler import AbstractSampler

NeighborhoodType = Literal['ball', 'circle', 'cube', 'square']
Interpolation = Literal['full', 'weighted', 'none']


class PointCloudDataset(Dataset):
    """A PyTorch Dataset to load a point cloud and query samples from this point cloud using a k-d tree.

    :param input_file_path: file path to point cloud file in a supported format (see utils.io.file_handler)
    :type input_file_path: pathlib.Path
    :param save_tree: whether to save trees to file or not
    :type save_tree: bool
    :param k: how many points are returned from a neighborhood query. When not enough points are inside the radius,
        points will be duplicated to reach k
    :type k: int
    :param neighborhood_type: Geometry of the neighborhood to be used:

        - `'ball'`: Retrieves points within a specified radius around the query point in a spherical
          geometry.
        - `'circle'`: Retrieves points whose xy-coordinates lie within a circle around the query point with the
          specified radius.
        - `'cube'`: Retrieves points within a cube of width 2 * radius and centered at the query point.
        - `'square'`: Retrieves points whose xy-coordinates lie within a rectangle with a width of 2 * radius and
          whose center is the query point.
    :type neighborhood_type: string
    :param neighborhood_sampling: Strategy for sampling points from a neighborhood. To construct the model inputs,
        all points within a volume with the specified search radius around the query point are retrieved,
        and a fixed number of points are then sampled from this set of points using the specified strategy.

        - `'random_sampling'`: k points are selected randomly.
        - `'farthest_point_sampling'`: k points are selected using farthest point sampling.
        - `'feature_based_sampling'`: A certain percentage of the k points is selected according to the highest value
          of a given feature. The remaining percentage of the k points is selected randomly.
        - `'stratified_sampling_point_count'`: We divide the said volume into a number of shells equal to the number
          of `k_percentages` given and sample the specified `k_percentages` from these shells, assigning
          higher values of k to the shells with higher point count.
        - `'stratified_sampling_concentric'`: The sampling volume is divided into a number of shells equal to
          the number of `k_percentages` given and the number of points to sample from each shell is determined
          based on the specified `k_percentages`, increasing the values of k from the outer to the inner shells.
        - `'stratified_sampling_curvature'`: The sampling volume is divided into a number of shells equal
          to the number of `k_percentages` given and the number of points to sample from each shell is determined
          based on the specified `k_percentages`, assigning higher values of k to the shells with higher mean
          curvature values. The indices inside the shells are also sorted according to their corresponding
          curvature values.
        - `'uniform_sampling'`: k points are selected by uniform sampling, whereby the space filled by the point
          cloud is divided into approximately k equally sized buckets and at least one point is taken from each
          filled bucket.
        - `'grid_sampling'`: k points are sampled using a combination of grid sampling and random sampling.
    :type neighborhood_sampling: string
    :param sampling_grid_size: Minimum grid size for `grid_sampling`.
    :type sampling_grid_size: float, optional
    :param sampling_feature: Feature to use for `feature_based_sampling`, if applicable.
    :type sampling_feature: string, optional
    :param stratified_sampling_strategy: Strategy to use for the `stratified_sampling` approach.
    :type stratified_sampling_strategy: string, optional
    :param k_percentages: String containing the percentages of k to be assigned to each shell. Only used for the
        `stratified_sampling` strategy.
    :type k_percentages: List[float], optional
    :param radius: radius for neighborhood to query
    :type radius: float
    :param cache_dir: folder path to folder where created k-d trees are saved. Defaults to `None`, which means that
        k-d trees are not saved.
    :type cache_dir: string, optional
    :param loss_weight_attribute: Name of a point attribute that should be used to weight the loss of each point.
        Defaults to `None`.
    :type loss_weight_attribute: string, optional
    :param sampling_weights: Dictionary mapping for each semantic class a weight according to the number of points in
        this class. See generate_metadata() in preprocess.py for more information.
    :type sampling_weights: dictionary
    :param transform: If true data augmentations are applied before returning the data for a neural network
    :type transform: bool
    :param known_columns: Columns to be used as model inputs. Defaults to `None`, which means that all columns are used
        as model inputs.
    :type known_columns: List[str].
    :param random_generator: If not `None`, this random number generator is used for data shuffling and data
        augmentation. Defaults to `None`.
    :type random_generator: numpy.random.Generator, optional
    """

    def __init__(self,
                 input_file_path: Path,
                 save_tree: bool,
                 k: int,
                 neighborhood_type: NeighborhoodType,
                 neighborhood_sampling: str,
                 radius: float,
                 sampling_grid_size: Optional[float] = None,
                 sampling_feature: Optional[str] = None,
                 stratified_sampling_strategy: Optional[str] = None,
                 k_percentages: Optional[List[float]] = None,
                 loss_weight_attribute: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 sampling_weights: Optional[Dict[int, float]] = None,
                 transform: bool = False,
                 known_columns: Optional[List[str]] = None,
                 random_generator: Optional[np.random.Generator] = None) -> None:
        self.random_generator = random_generator

        self.reader = FileReaderManager()
        self.file_name = str(Path(input_file_path).stem)
        self.dataset: Union[pd.DataFrame, np.ndarray] = self.reader.read(input_file_path).data_shifted()
        self.point_sampler: AbstractSampler

        self.point_weights: Optional[np.ndarray]
        if loss_weight_attribute is not None:
            if loss_weight_attribute not in self.dataset.columns:
                raise ValueError(f'{loss_weight_attribute} was specified as loss weight but is not included in the '
                                 f'point cloud {self.file_name}')
            self.point_weights = self.dataset[loss_weight_attribute].values  # type: ignore[assignment]
        else:
            self.point_weights = None

        if known_columns is not None:
            self.known_columns = known_columns
            unknown_columns = [x for x in self.dataset.columns if x not in self.known_columns and x != "semclassid"]
            self.dataset = self.dataset.drop(columns=unknown_columns)
        else:
            self.known_columns = list(self.dataset.columns.astype(str))

        self.neighborhood_type = neighborhood_type
        self.neighborhood_sampling = neighborhood_sampling

        self.save_tree = save_tree
        self.cache_dir = cache_dir
        self._get_output_file_path(input_file_path)
        self._get_tree()

        if "semclassid" in self.dataset.columns:
            labels = self.dataset["semclassid"].astype(np.int32)
            self.dataset = self.dataset.drop(columns=['semclassid'])
        else:
            labels = pd.Series(np.zeros(self.dataset.shape[0], dtype=np.int32))

        self.normal_index = self.dataset.columns.get_loc('nx') if 'nx' in self.dataset.columns else -1
        self.columns = self.dataset.columns
        self.dataset = self.dataset.values

        if sampling_weights is not None:
            label_class_ids = np.unique(labels)
            weight_class_ids = np.array(list(sampling_weights.keys()))

            class_id_diff = np.setdiff1d(label_class_ids, weight_class_ids)
            if len(class_id_diff) > 0:
                raise ValueError(f'The dataset contains class IDs that are not included in the metadata file: '
                                 f'{class_id_diff}.')

            self.sampling_weights = np.array(labels.map(sampling_weights).values)
        else:
            self.sampling_weights = np.ones(len(labels))
        self.labels = np.array(labels.values)

        self.transform = transform
        self.k = k
        self.radius = radius

        if "grid_sampling" in self.neighborhood_sampling:
            if sampling_grid_size is None:
                raise ValueError('`sampling_grid_size` argument was not specified.')
            self.sampling_grid_size = sampling_grid_size

        # Validating that the sampling feature is not none for feature-based or stratified sampling with feature-based
        # strategy.
        if "feature_based_sampling" in self.neighborhood_sampling:
            self.sampling_feature = self._validate_sampling_feature(sampling_feature)

        # Validating that k_percentages has the correct format and obtaining the
        # said percentages as a numpy array.
        if "stratified_sampling" in self.neighborhood_sampling:
            if stratified_sampling_strategy is None:
                raise ValueError('`stratified_sampling_strategy` argument was not specified.')
            else:
                self.stratified_sampling_strategy = stratified_sampling_strategy
                if "feature_based" in self.stratified_sampling_strategy:
                    self.sampling_feature = self._validate_sampling_feature(sampling_feature)
            # Validating the correct format.
            self.k_percentages = np.array(k_percentages).astype(float)
            if np.sum(self.k_percentages) != 100.0:
                raise ValueError('The `k_percentages` should add to a total of 100.0')

        if k >= self.__len__():
            logging.info('Point cloud {} only contains {} points, less than the specified neighborhood size of {}. '
                         'Sampling this file might lead to bad results.'.format(input_file_path, self.__len__(), k))

        self.initialize_sampler(self.neighborhood_sampling)

    def initialize_sampler(self, sampler_type: str) -> None:
        """Initializes the dataset's point sampler.

        :param sampler_type: Strategy for sampling points from a neighborhood.
        :type sampler_type: String
        """
        if sampler_type == "grid_sampling":
            self.point_sampler = GridSampler(self.k, self.sampling_grid_size, generator=self.random_generator)
        else:
            # default to RandomSampler for now
            self.point_sampler = RandomSampler(self.k, self.random_generator)

    def __len__(self) -> int:
        """Overridden to comply with the PyTorch Dataset interface. Returns the number of
        points in the point cloud.

        :return: number of points in the data
        :rtype: number"""
        return len(self.dataset)

    def _validate_sampling_feature(self, sampling_feature: Optional[str]) -> str:
        """ Validates that the `sampling_feature` value exists and matches an existing column in
        the dataset.

        :param sampling_feature: feature name to be checked.
        :type sampling_feature: string

        :return: the sampling feature string.
        :rtype: string
        """
        if sampling_feature is None:
            raise ValueError('`sampling_feature` argument required for a feature-based sampling approach.')
        else:
            # Check that the feature exists in the dataset.
            if sampling_feature not in self.columns.tolist():
                raise ValueError('`sampling_feature` argument is not present in the dataset.')
            else:
                return sampling_feature

    def _scale_data(self, query_point: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
        """Scales a given neighborhood according to the `neighborhood_type` in use.

        :param query_point: point from which the neighborhood was queried.
        :type query_point: numpy.ndarray
        :param neighborhood: Points inside the neighborhood to be scaled.
        :type neighborhood: numpy.ndarray

        :return: the scaled neighborhood.
        :rtype: numpy.ndarray
        """

        if self.neighborhood_type in ["circle", "square"]:
            scaled_neighborhood = translate_scale_xy(neighborhood, center_point=query_point,
                                                     scale_factor=1 / self.radius).astype(np.float32)
        else:
            scaled_neighborhood = translate_scale_xyz(neighborhood, center_point=query_point,
                                                      scale_factor=1 / self.radius).astype(np.float32)

        return scaled_neighborhood

    def get_neighborhood(self,
                         idx: int,
                         prediction: Optional[bool] = False,
                         interpolation: Interpolation = 'full') -> Dict[str,
                                                                        Union[np.ndarray, List[np.ndarray]]]:
        """Obtains the points in the neighborhood of a given point and samples a fixed number of points
        from it according to the strategy specified in `self.neighborhood_type` and `self.neighborhood_sampling`.
        Note that we only do subsampling if the amount of points within a specified radius is greater
        than the value of k.
        If k = -1, all the points within the radius are taken as well.

        :param idx: index of the point for which the neighborhood should be queried
        :type idx: integer
        :param prediction: boolean flag indicating if this method is being called during the prediction process.
            Used as there are additional steps not needed during training.
        :type prediction: bool, optional
        :param interpolation: interpolation strategy to be used for unpredicted points. Defaults to `full`.
        :type extrapolaton: str, optional

        :return: Dictionary containig the data of the neighborhood. The dictionary always contains the following
            entries:

            - `points`: The coordinates and features of the points of shape `(N, 3 + D)`, where
              `N = number of points sampled from the neighborhood`, and `D = number of feature channels per point`.
            - `point_indices`: The indices of the points within the complete dataset of shape `(N)`.
            - `is_duplicate`: A boolean mask indicating which points are duplicate points of shape `(N)`.
              if `prediction = False`, the following additional entry is provided:
            - `labels`: The labels of the points of shape `(N)`. If `prediction = False` and `self.point_weights` is not
              `None`, the following additional entry is provided:
            - `point_weights`: Point weights for weighting the loss of shape `(N)`.
              If `prediction = True`, the following additional entries are provided:
            - `interpolation_points`: The coordinates and features of the points discarded during sampling points from
              the neighborhood of shape `(N', 3 + D)` where `N' = number of points discarded during sampling`.
            - `interpolation_indices`: The indices of the points discarded during sampling points from the neighborhood
              of shape `(N')`.
        :rtype: Dict[str, Union[np.ndarray, List[np.ndarray]]]
        """

        assert isinstance(self.dataset, np.ndarray)

        # Getting neighborhood.
        point, neighborhood_indices = self.query_neighborhood(idx, self.radius)

        if self.k == -1:
            # If no k value is specified, we take all the points in the neighborhood as-is.
            k_indices = neighborhood_indices
            is_duplicate = np.zeros(len(neighborhood_indices), dtype=bool)
            interpolation_indices = np.empty(0, dtype=int)
            interpolation_points = np.empty(0, dtype=np.float32)
        else:
            # Potential cases when doing radius-based sampling.
            if len(neighborhood_indices) <= self.k:
                # If a value for k is specified but there are less than k points inside the given
                # radius, we duplicate as necessary and take all the points.
                k_indices, is_duplicate = repeat_if_necessary(neighborhood_indices, self.k)
                # In this case there are not excluded points.
                interpolation_indices = np.empty(0, dtype=int)
                interpolation_points = np.empty(0, dtype=np.float32)
            else:
                # If we have more than k points within the specified radius, we utilize the specified
                # subsampling strategy.
                if "farthest_point_sampling" in self.neighborhood_sampling:
                    points = self.dataset[neighborhood_indices]
                    k_indices = farthest_point_sampling_cluster(neighborhood_indices, points[:, :3],
                                                                self.k, generator=self.random_generator)
                elif "feature_based_sampling" in self.neighborhood_sampling:
                    column_names = self.columns.tolist()
                    column_idx = column_names.index(self.sampling_feature)
                    point_feature_vals = self.dataset[neighborhood_indices][:, column_idx]
                    k_indices = feature_based_sampling(neighborhood_indices,
                                                       point_feature_vals,
                                                       self.k,
                                                       generator=self.random_generator)
                elif "stratified_sampling" in self.neighborhood_sampling:
                    # Getting the point indices within the given shells.
                    shell_indices = self.query_neighborhood_shells(idx,
                                                                   self.radius,
                                                                   self.k_percentages,
                                                                   neighborhood_indices)
                    if "concentric" in self.stratified_sampling_strategy or \
                            "point_count" in self.stratified_sampling_strategy:
                        points = None
                        feature_idx = None
                    else:
                        points = [self.dataset[aux_indices] for aux_indices in shell_indices]
                        sampling_cols = self.columns.tolist()
                        feature_idx = sampling_cols.index(self.sampling_feature)
                    # Getting the selected indices for the feature-based extension.
                    # Calculating the radii of the two smaller regions.
                    k_indices = stratified_sampling(shell_indices,
                                                    self.k,
                                                    self.k_percentages,
                                                    self.stratified_sampling_strategy,
                                                    generator=self.random_generator,
                                                    shell_points=points,
                                                    feature_idx=feature_idx)
                elif "uniform_sampling" in self.neighborhood_sampling:
                    points = self.dataset[neighborhood_indices]
                    k_indices = uniform_sampling(points, neighborhood_indices, self.k,
                                                 generator=self.random_generator)
                else:
                    # We default to using the pre-initialized sampler.
                    k_indices = self.point_sampler(self.dataset[neighborhood_indices], neighborhood_indices)
                is_duplicate = np.zeros(len(k_indices), dtype=bool)

                # Constructing neighborhood with the obtained indices.
                if idx not in k_indices:
                    k_indices[-1] = idx  # Making sure we include the seed index in the neighborhood.

                if prediction:
                    if interpolation == 'full':
                        # Obtaining the not chosen indices. We do this by computing the set difference
                        # between the full neighborhood and the indices that were selected by
                        # the sampling method.
                        interpolation_indices = np.setdiff1d(neighborhood_indices, k_indices, assume_unique=True)
                        interpolation_points = self.dataset[interpolation_indices]
                    elif interpolation == 'none':
                        interpolation_indices = np.empty(0, dtype=int)
                        interpolation_points = np.empty(0, dtype=np.float32)

        neighborhood = self.dataset[k_indices]

        neighborhood = self._scale_data(point, neighborhood)

        # Applying transformations if necessary.
        if self.transform:
            neighborhood = jitter(neighborhood, radius=self.radius, generator=self.random_generator)
            neighborhood = random_scale(neighborhood, generator=self.random_generator)
            neighborhood = random_rotate(neighborhood, self.normal_index, generator=self.random_generator)
            neighborhood = shift(neighborhood, generator=self.random_generator)

        neighborhood_data = {
            'points': neighborhood.astype(np.float32),
            'point_indices': k_indices.astype(np.int64),
            'is_duplicate': is_duplicate
        }
        # If we are in the prediction step, we need to return the not chosen indices and points as well.
        if prediction:
            if len(interpolation_points) > 0:
                interpolation_points = self._scale_data(point, interpolation_points)
            neighborhood_data['interpolation_points'] = interpolation_points.astype(np.float32)
            neighborhood_data['interpolation_indices'] = interpolation_indices.astype(np.int64)
        else:
            neighborhood_data['labels'] = self.labels[k_indices].astype(np.int64)

            if self.point_weights is not None:
                neighborhood_data['point_weights'] = self.point_weights[k_indices].astype(np.float32)

        return neighborhood_data

    def __getitem__(self, idx) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Overridden to comply with the PyTorch Dataset interface. Returns a sample for the given index by querying
        the internal data structure at the given index.

        :param idx: index to which a sample (point neighborhood) should be returned
        :type idx: integer

        :return: Dictionary containig the data of the neighborhood. The dictionary always contains the following
            entries:
            - | `points`: The coordinates and features of the points of shape `(N, 3 + D)`, where
              | `N = number of points sampled from the neighborhood`, and `D = number of feature channels per point`.
            - | `point_indices`: The indices of the points within the complete dataset of shape `(N)`.
            - | `is_duplicate`: A boolean mask indicating which points are duplicate points of shape `(N)`.
            - | `labels`: The labels of the points of shape `(N)`.
            If `self.point_weights` is not `None`, the following additional entry is provided:
            - | `point_weights`: Point weights for weighting the loss of shape `(N)`.
        :rtype: Dict[str, Union[np.ndarray, List[np.ndarray]]]
        """

        return self.get_neighborhood(idx)

    def id(self) -> str:
        """
        :return: ID of the dataset.
        :rtype: string.
        """

        hash_string = utils.hash_dataset(self.dataset)

        return self.file_name + '_' + hashlib.md5(hash_string.encode('utf-8')).hexdigest()[:10]

    def _get_output_file_path(self, input_file_path: Path) -> None:
        """Determines the tree output file name by hashing part of the dataset.

           :param input_file_path: File path to point cloud file.
           :type input_file_path: str
           :return: nothing
           """

        if self.cache_dir is None:
            self.output_file_path = None
        else:
            assert isinstance(self.dataset, pd.DataFrame)

            hash_string = ''.join(str(x) for x in [self.dataset.shape, self.dataset.head(2), self.dataset.tail(2)])
            hash_id = hashlib.md5(hash_string.encode('utf-8')).hexdigest()[:10]
            file_name = Path(input_file_path).name + "_" + hash_id + "_" + self.neighborhood_type + ".bin"
            self.output_file_path = Path(self.cache_dir) / 'trees' / file_name

    def query_neighborhood(self, index: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Queries the configured neighborhood by querying the k-d tree at the given index and
        returns the point at this index and the indices of the neighbors.

        :param index: index of the point for which the neighborhood should be queried
        :type index: integer
        :param radius: search radius for spherical/cylindrical search.
        :type radius: float

        :return: the point and the indices of the neighborhood points, the labels of the k neighborhood points
        :rtype: tuple of numpy.ndarray, numpy.ndarray
        """

        assert isinstance(self.dataset, np.ndarray)

        point = self.dataset[index]

        if self.neighborhood_type in ["circle", "square"]:
            query_point = np.ascontiguousarray(point[:2])
        else:
            query_point = np.ascontiguousarray(point[:3])

        p_norm = np.inf if self.neighborhood_type in ["cube", "square"] else 2
        indices = self.tree.query_ball_point(query_point, radius, workers=1, return_sorted=False, p=p_norm)

        return point, np.array(indices)

    def query_neighborhood_shells(self, index: int,
                                  rad_big: float,
                                  k_percentages: np.ndarray,
                                  indices_big: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Divides a given neighborhood in shells (as many as `k_percentages` are given),
        given the radius of the outermost shell and returns the indexes of the points
        within each of the shells.

        :param index: index of the point for which the neighborhoods should be queried.
        :type index: integer
        :param rad_big: biggest search radius/radius of the biggest shell to be queried.
        :type radius_big: float
        :param k_percentages: Percentages of k to be assigned to each shell.
            As many shells as k percentages are given will be generated.
        :type k_percentages: numpy.ndarray
        :param indices_big: indices of the points within the outermost radius.
        :type indices_big: numpy.ndarray, optional

        :return: list of numpy.ndarrays containing the indexes within the specified shells.
        :rtype: List[numpy.ndarray]
        """
        num_shells = len(k_percentages)
        max_shell_idx = num_shells

        # If the biggest neighborhood is not given, we look for it as well.
        if indices_big is None:
            # By doing this, we will obtain and calculate the biggest
            # radius in the following loop, otherwise it will be skipped.
            max_shell_idx += 1

        # Calculating the radius sizes.
        radii = []
        for k in range(1, max_shell_idx):
            current_radius = k * (rad_big / num_shells)
            radii.append(current_radius)

        # Getting the indices of the points inside all the volumes.
        all_neighborhoods = []
        for radius in radii:
            _, current_indices = self.query_neighborhood(index, radius)
            all_neighborhoods.append(current_indices)
        # If indices_big was given, we just append it at the end.
        if indices_big is not None:
            all_neighborhoods.append(indices_big)

        # Computing the set differences to remove overlapping regions.
        # We go backwards as we want to start with the biggest regions,
        # and we do not want to remove anything from the smallest one.
        for j in reversed(range(1, len(all_neighborhoods))):
            all_neighborhoods[j] = np.setdiff1d(all_neighborhoods[j],
                                                all_neighborhoods[j - 1],
                                                assume_unique=True)
        return all_neighborhoods

    def _load_tree(self) -> None:
        """Loads a saved k-d tree from the output file path and stores it in self.tree.

        :return: nothing
        """
        assert self.output_file_path is not None

        with open(str(self.output_file_path), 'rb') as f:
            self.tree = pickle.load(f)

    def _create_tree(self) -> None:
        """Creates a k-d tree from the point cloud and saves this tree to the internal output path.

        :return: nothing
        """

        assert isinstance(self.dataset, pd.DataFrame)

        logging.debug("Creating tree.")

        if self.neighborhood_type in ["circle", "square"]:
            coords = self.dataset[['x', 'y']].values
        else:
            coords = self.dataset[['x', 'y', 'z']].values

        self.tree = KDTree(coords, balanced_tree=False, leafsize=1000)
        if self.save_tree and self.output_file_path is not None:
            self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
            logging.debug("Saving tree at " + str(self.output_file_path))
            with open(str(self.output_file_path), 'wb') as f:
                pickle.dump(self.tree, f)

    def _get_tree(self) -> None:
        """Checks if a k-d tree already exists at the internal output path. If this is the case, this tree is
        loaded from disk, otherwise a new tree is created and potentially saved.

        :return: nothing
        """
        if self.output_file_path is not None and self.output_file_path.is_file():
            self._load_tree()
        else:
            self._create_tree()


class PredictionPointCloudDataset(IterableDataset):
    """A PyTorch :code:`IterableDataset` to sample all points from a point cloud. It makes use of a
    :code:`PointCloudDataset` for sampling the points and is meant to be used for prediction only. Note: as it does not
    implement a :code:`__len__` method, it does not work out of the box with progress bars.

    :param input_file_path: file path to point cloud file in a supported format (see utils.io.file_handler)
    :type input_file_path: pathlib.Path
    :param k: determines how many points are sampled from a neighborhood. Setting it to -1 is strongly recommended for
        performance and convergence.
    :type k: int
    :param neighborhood_type: Geometry of the neighborhood to be used:

        - `'ball'`: Retrieves points within a specified radius around the query point in a spherical
          geometry.
        - `'circle'`: Retrieves points whose xy-coordinates lie within a circle around the query point with the
          specified radius.
        - `'cube'`: Retrieves points within a cube of width 2 * radius and centered at the query point.
        - `'square'`: Retrieves points whose xy-coordinates lie within a square with a width of 2 * radius and
          whose center is the query point.
    :type neighborhood_type: string
    :param neighborhood_sampling: Strategy for sampling points from a neighborhood. To construct the model inputs,
        all points within a volume with the specified search radius around the query point are retrieved,
        and a fixed number of points are then sampled from this set of points using the specified strategy.

        - `'random_sampling'`: k points are selected randomly.
        - `'farthest_point_sampling'`: k points are selected using farthest point sampling.
        - `'feature_based_sampling'`: A certain percentage of the k points is selected according to the highest value
          of a given feature. The remaining percentage of the k points is selected randomly.
        - `'stratified_sampling_point_count'`: We divide the said volume into a number of shells equal to the number
          of `k_percentages` given and sample the specified `k_percentages` from these shells, assigning
          higher values of k to the shells with higher point count.
        - `'stratified_sampling_concentric'`: The sampling volume is divided into a number of shells equal to
          the number of `k_percentages` given and the number of points to sample from each shell is determined
          based on the specified `k_percentages`, increasing the values of k from the outer to the inner shells.
        - `'stratified_sampling_curvature'`: The sampling volume is divided into a number of shells equal
          to the number of `k_percentages` given and the number of points to sample from each shell is determined
          based on the specified `k_percentages`, assigning higher values of k to the shells with higher mean
          curvature values. The indices inside the shells are also sorted according to their corresponding
          curvature values.
        - `'grid_sampling'`: k points are sampled using a combination of grid sampling and random sampling.
        - `'uniform_sampling'`: k points are selected by uniform sampling, whereby the space filled by the point
          cloud is divided into approximately k equally sized buckets and at least one point is taken from each
          filled bucket.
    :type neighborhood_sampling: string
    :param sampling_grid_size: Minimum grid size for `grid_sampling`.
    :type sampling_grid_size: float, optional
    :param sampling_feature: Feature to use for `feature_based_sampling`, if applicable.
    :type sampling_feature: string, optional
    :param stratified_sampling_strategy: Strategy to use for the `stratified_sampling` approach.
    :type stratified_sampling_strategy: string, optional
    :param k_percentages: String containing the percentages of k to be assigned to each shell. Only used for the
        `stratified_sampling` strategy.
    :type k_percentages: List[float], optional
    :param radius: radius of the queried neighborhood
    :type radius: float
    :param draw_at_once: determined how many neighborhoods are drawn before updating the internal index structure.
        Higher values are computationally less expensive, but take longer to converge and return more duplicates.
    :type draw_at_once: int
    :param interpolation: interpolation strategy to be used for unpredicted points. Defaults to `full`.
    :type extrapolaton: str, optional
    :param cache_dir: folder path to folder where created k-d trees are saved. Defaults to `None`, which means that
        k-d trees are not saved.
    :type cache_dir: string, optional
    :param known_columns: Columns to be used as model inputs. Defaults to `None`, which means that all columns are used
        as model inputs.
    :type known_columns: List[str].
    :param seed: Random seed to be used for seeding data shuffling. Defaults to `None`, which means that data shuffling
        is not seeded.
    :type seed: integer, optional
    """

    def __init__(self,
                 input_file_path: Path,
                 k: int,
                 neighborhood_type: NeighborhoodType,
                 neighborhood_sampling: NeighborhoodSampling,
                 radius: float,
                 draw_at_once: int,
                 interpolation: Interpolation = 'full',
                 sampling_grid_size: Optional[float] = None,
                 sampling_feature: Optional[str] = None,
                 stratified_sampling_strategy: Optional[str] = None,
                 k_percentages: Optional[List[float]] = None,
                 cache_dir: Optional[str] = None,
                 known_columns: Optional[List[str]] = None,
                 seed: Optional[int] = None) -> None:
        self.dataset = PointCloudDataset(input_file_path, False, k, neighborhood_type,
                                         neighborhood_sampling,
                                         radius,
                                         sampling_grid_size=sampling_grid_size,
                                         sampling_feature=sampling_feature,
                                         stratified_sampling_strategy=stratified_sampling_strategy,
                                         k_percentages=k_percentages,
                                         cache_dir=cache_dir,
                                         known_columns=known_columns)
        self.hit = torch.zeros(len(self.dataset), requires_grad=False)
        self.hit.share_memory_()
        self.lock = Lock()
        self.draw_at_once = draw_at_once
        self.sampling_indices: Deque[int] = deque()
        self.seed = seed
        self.numpy_random_generator = None
        self.torch_random_generator = None
        self.interpolation = interpolation

    def __iter__(self):
        self.numpy_random_generator = np.random.default_rng(self.seed)
        self.dataset.random_generator = self.numpy_random_generator
        self.torch_random_generator = None
        if self.seed is not None:
            self.torch_random_generator = torch.Generator()
            self.torch_random_generator.manual_seed(self.seed)

        with self.lock:
            self.hit *= 0
        self.sampling_indices = deque()

        return self

    def __next__(self) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Takes a not-yet-sampled point as a center for a neighborhood and samples said neighborhood.

        :return: Dictionary containig the data of the neighborhood. The dictionary contains the following entries:
            - | `points`: The coordinates and features of the points of shape `(N, 3 + D)`, where
              | `N = number of points sampled from the neighborhood`, and `D = number of feature channels per point`.
            - | `point_indices`: The indices of the points within the complete dataset of shape `(N)`.
            - | `is_duplicate`: A boolean mask indicating which points are duplicate points of shape `(N)`.
            - | `interpolation_points`: The coordinates and features of the points discarded during sampling points from
              | the neighborhood of shape `(N', 3 + D)` where `N' = number of points discarded during sampling`.
            - | `interpolation_indices`: The indices of the points discarded during sampling points from
              | the neighborhood of shape `(N')`.
        :rtype: Dict[str, Union[np.ndarray, List[np.ndarray]]]
        """
        if len(self.sampling_indices) == 0:
            current_indices = torch.where(self.hit == 0)[0]
            if len(current_indices) == 0:
                raise StopIteration
            index_locations = torch.randint(len(current_indices), size=(self.draw_at_once,),
                                            generator=self.torch_random_generator)
            self.sampling_indices.extend(current_indices[index_locations].numpy())

        index = self.sampling_indices.pop()
        neighborhood_data = self.dataset.get_neighborhood(index, prediction=True, interpolation=self.interpolation)

        self.update_hit(np.concatenate((neighborhood_data['point_indices'],
                                       neighborhood_data['interpolation_indices'])))

        return neighborhood_data

    def update_hit(self, point_indices: np.ndarray) -> None:
        with self.lock:
            self.hit[point_indices] += 1  # type: ignore [index]

    def num_points(self) -> int:
        return len(self.dataset)

    def num_features(self) -> int:
        return len(self.dataset.columns)

    def id(self):
        """
        :return: ID of the dataset.
        :rtype: string.
        """

        return 'prediction_' + self.dataset.id()


def identity_collate(batch: List[Any]) -> Any:
    # dummy function needed for multiprocessing
    return batch


def collate_prediction_batch(
    batch: List[Dict[str, Union[np.ndarray, List[np.ndarray]]]]
) -> Dict[str, Union[torch.Tensor, List[np.ndarray]]]:
    """
    Collates batches returned by the `PredictionPointCloudDataset`.

    :param batch: List of dictionaries representing the bactch items to be collated. Except for the entries
        `interpolation_indices`, and `interpolation_points`, all batch items are expected to contain the same number of
        points.
    :type batch: List[Dict[str, Union[np.ndarray, List[np.ndarray]]]]

    :return: Collated batch. Except for the entries `interpolation_indices` and `interpolation_points`, the batch
        items are packed into PyTorch tensors of shape `(B, N, ...)`, where `B = batch size`, and
        `N = number of points`. `interpolation_indices`, and `interpolation_points` are packed into lists of
        numpy arrays.
    :rytpe: Dict[str, Union[torch.Tensor, List[np.ndarray]]]
    """
    interpolation_indices = [batch_item.pop('interpolation_indices') for batch_item in batch]
    interpolation_points = [torch.tensor(batch_item.pop('interpolation_points')) for batch_item in batch]

    collated_batch = default_collate(batch)
    collated_batch['interpolation_indices'] = interpolation_indices
    collated_batch['interpolation_points'] = interpolation_points

    return collated_batch


def get_prediction_dataloader(batch_size: int,
                              k: int,
                              known_columns: List[str],
                              neighborhood_sampling: NeighborhoodSampling,
                              neighborhood_type: NeighborhoodType,
                              num_workers: int,
                              path: Path,
                              radius: float,
                              interpolation: Interpolation = 'full',
                              cache_dir: Optional[str] = None,
                              sampling_grid_size: Optional[float] = None,
                              sampling_feature: Optional[str] = None,
                              stratified_sampling_strategy: Optional[str] = None,
                              k_percentages: Optional[List[float]] = None,
                              seed: Optional[int] = None) -> DataLoader:
    """Creates a PyTorch DataLoader from the given file, which can be used for prediction.

    :param batch_size: number of neighborhoods to return at once
    :type batch_size: int
    :param k: number of points to sample from a neighborhood.When not enough points are inside the radius,
        points will be duplicated to reach k.
    :type k: int
    :param known_columns: Columns to be used as model inputs. Defaults to `None`, which means that all columns are used
        as model inputs.
    :type known_columns: List[str].
    :param neighborhood_sampling: Strategy for sampling points from a neighborhood. To construct the model inputs,
        all points within a volume with the specified search radius around the query point are retrieved,
        and a fixed number of points are then sampled from this set of points using the specified strategy.

        - `'random_sampling'`: k points are selected randomly.
        - `'farthest_point_sampling'`: k points are selected using farthest point sampling.
        - `'feature_based_sampling'`: A certain percentage of the k points is selected according to the highest value
          of a given feature. The remaining percentage of the k points is selected randomly.
        - `'stratified_sampling_point_count'`: We divide the said volume into a number of shells equal to the number
          of `k_percentages` given and sample the specified `k_percentages` from these shells, assigning
          higher values of k to the shells with higher point count.
        - `'stratified_sampling_concentric'`: The sampling volume is divided into a number of shells equal to
          the number of `k_percentages` given and the number of points to sample from each shell is determined
          based on the specified `k_percentages`, increasing the values of k from the outer to the inner shells.
        - `'stratified_sampling_curvature'`: The sampling volume is divided into a number of shells equal
          to the number of `k_percentages` given and the number of points to sample from each shell is determined
          based on the specified `k_percentages`, assigning higher values of k to the shells with higher mean
          curvature values. The indices inside the shells are also sorted according to their corresponding
          curvature values.
        - `'grid_sampling'`: k points are sampled using a combination of grid sampling and random sampling.
        - `'uniform_sampling'`: k points are selected by uniform sampling, whereby the space filled by the point
          cloud is divided into approximately k equally sized buckets and at least one point is taken from each
          filled bucket.
    :type neighborhood_sampling: string
    :param neighborhood_type: Geometry of the neighborhood to be used:

        - `'ball'`: Retrieves points within a specified radius around the query point in a spherical
          geometry.
        - `'circle'`: Retrieves points whose xy-coordinates lie within a circle around the query point with the
          specified radius.
        - `'cube'`: Retrieves points within a cube of width 2 * radius and centered at the query point.
        - `'square'`: Retrieves points whose xy-coordinates lie within a rectangle with a width of 2 * radius and
          whose center is the query point.
    :type neighborhood_type: string
    :param num_workers: number of additional threads used by the DataLoader for loading data
        (0 means no parallelism)
    :type num_workers: integer
    :param path: path to a point cloud
    :type path: string
    :param radius: radius the queried neighborhoods should have
    :type radius: float
    :param interpolation: interpolation strategy to be used for unpredicted points. Defaults to `full`.
    :type extrapolaton: str, optional
    :param cache_dir: folder path to folder where created k-d trees are saved. Defaults to `None`, which means that
        k-d trees are not saved.
    :type cache_dir: string, optional
    :param sampling_grid_size: Minimum grid size for `grid_sampling`.
    :type sampling_grid_size: float, optional
    :param sampling_feature: Feature to use for `feature_based_sampling`, if applicable.
    :type sampling_feature: string, optional
    :param ss_strategy: Strategy to use for the `stratified_sampling` approach.
    :type stratified_sampling_strategy: string, optional
    :param k_percentages: String containing the percentages of k to be assigned to each shell. Only used for the
        `stratified_sampling` strategy.
    :type k_percentages: List[float], optional
    :param seed: Random seed to be used for seeding data shuffling. Defaults to `None`, which means that data shuffling
        is not seeded.
    :type seed: integer, optional
    :return: PyTorch DataLoader for the dataset.
    :rtype: torch.utils.data.DataLoader.DataLoader
    """

    dataset = PredictionPointCloudDataset(
        input_file_path=path,
        k=k,
        neighborhood_type=neighborhood_type,
        neighborhood_sampling=neighborhood_sampling,
        sampling_grid_size=sampling_grid_size,
        sampling_feature=sampling_feature,
        stratified_sampling_strategy=stratified_sampling_strategy,
        k_percentages=k_percentages,
        radius=radius,
        draw_at_once=10,
        interpolation=interpolation,
        known_columns=known_columns,
        cache_dir=cache_dir,
        seed=seed
    )

    if seed is not None:
        main_random_generator = torch.Generator()
        main_random_generator.manual_seed(seed)
    else:
        main_random_generator = None

    if k == -1:
        collate_fn = identity_collate
    else:
        collate_fn = collate_prediction_batch

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=utils.worker_init_fn,
        generator=main_random_generator
    )

    return loader
