__all__ = ['get_dataloaders', 'MultiPointCloudLoader']

from concurrent import futures
import math
from pathlib import Path
import sys
from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, RandomSampler, WeightedRandomSampler
from tqdm import tqdm

from .file_handler import FileReaderManager
from .general_utils import worker_init_fn, get_length, hash_dataset
from .point_cloud_dataset import PointCloudDataset, NeighborhoodType
from .sampling_utils import NeighborhoodSampling
from .multi_point_cloud_sampler import MultiPointCloudSampler, SamplingWeight

SamplingStrategy = Literal['class_weights', 'uniform']
LoadingCallback = Callable[[DataLoader, str], None]


class MultiPointCloudLoader:
    """Loader for multiple point cloud files, keeping only one file in memory at a time.
    Points are sampled from files proportional to how many points each file contains, e.g.
    if file_1 contains 1000 points and file_2 contains 500 points, each epoch will contain 2/3 * n
    points from file_1.

    :param files: list of filenames as strings, e.g. ["data/train_file1.h5", "data/train_file2.h5"]
    :type files: List[pathlib.Path]
    :param cache_dir: Path to folder to be used for caching files.
    :type cache_dir: string
    :param class_ids: List of all class IDs used in the point cloud datasets.
    :type class_ids: List[integer]
    :param save_trees: boolean determining whether cKD trees should be saved.
    :type save_trees: boolean
    :param batch_size: size of each batch
    :type batch_size: integer
    :param k: number of points drawn for each neighborhood
    :type k: integer
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
    :param radius: radius for the neighborhood queries in meters
    :type radius: float
    :param n: Number of samples to be drawn. Defaults to `-1`, which means that the number of samples to draw is
        calculated using the specified `length_divider` value.
    :type n: integer, optional
    :param num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main
        process. Number is forwarded to PyTorch DataLoader. Process creation time might lead to bad performance when
        using high number of workers.
    :type num_workers: num_workers
    :param loss_weight_attribute: Name of a point attribute that should be used to weight the loss of each point.
        Defaults to `None`.
    :type loss_weight_attribute: string, optional
    :param transform: boolean determining whether random rotate, jitter, etc. are applied. Should be true for train and
        false for test
    :type transform: boolean
    :param n: integer that determines number of points drawn in total. Setting it to -1 approximates sampling points
        such that each point is seen once
    :type n: integer
    :param known_columns: Columns to be used as model inputs. Defaults to `None`, which means that all columns are used
        as model inputs.
    :type known_columns: List[str].
    :param load_files_async: Whether the next point cloud file should be loaded asynchronously while iterating over the
        current one. Enabling this option improves loading speed but increases the dataloaders memory requirements.
        Defaults to `False`.
    :type load_files_async: bool, optional
    :param on_load_new_file: Callback to be called when a new point cloud file is loaded.
    :type on_load_new_file: Callable[[Union[DataLoader, RadiusNeighborhoodDataLoader], str], None], optional
    :param centroid_sampling: To obtain inputs for deep learning models, small-scale point clouds are sampled
        from the large-scale input point clouds. Each small-scale point cloud is obtained by selecting a centroid point
        from the large-scale point cloud and retrieving a certain point neighborhood of the centroid (e.g., a spherical
        neighborhood). The `centroid_sampling` parameter controls how centroids are sampled. The available
        options are: `'class_weights'` | `'uniform'`.

        - `'class_weights'`: Sampling of centroid points is weighted by the inverse class frequencies.
        - `'uniform'`: All points have the same probability of being sampled as a centroid point.
    :type centroid_sampling: string, optional
    :param centroid_sampling_weight: How classes should be weighted in sampling: `'unweighted'` | `'square_root'` |
        `'balanced'`.

        - `'unweighted'`: Points are drawn randomly without considering class labels.
        - `'square_root'`: Points are assigned sampling weights based on their class label. The sampling weights are
          chosen so that the class frequencies of the sampled points are (approximately) proportional to the square
          root of the class frequencies before sampling.
        - `'balanced'`: Points are assigned sampling weights based on their class label. The sampling weights are
          chosen so that (approximately) the same number of points is drawn from each class.

        Only used if `centroid_sampling_weight` is set to `'class_weights'`. Defaults to `'square_root'`.
    :type centroid_sampling_weight: string, optional
    :param seed: Random seed to be used for seeding data shuffling. Defaults to `None`, which means that data shuffling
        is not seeded.
    :type seed: integer, optional
    """

    def __init__(self,
                 files: List[Path],
                 cache_dir: str,
                 class_ids: List[int],
                 save_trees: bool,
                 batch_size: int,
                 k: int,
                 neighborhood_type: NeighborhoodType,
                 neighborhood_sampling: NeighborhoodSampling,
                 radius: float,
                 length_divider: int,
                 loss_weight_attribute: Optional[str] = None,
                 sampling_grid_size: Optional[float] = None,
                 sampling_feature: Optional[str] = None,
                 stratified_sampling_strategy: Optional[str] = None,
                 k_percentages: Optional[List[float]] = None,
                 n: int = -1,
                 num_workers: int = 0,
                 transform: bool = True,
                 known_columns: Optional[List[str]] = None,
                 load_files_async: bool = False,
                 on_load_new_file: Optional[LoadingCallback] = None,
                 centroid_sampling: SamplingStrategy = 'class_weights',
                 centroid_sampling_weight: SamplingWeight = 'square_root',
                 seed: Optional[int] = None) -> None:
        self.centroid_sampling = centroid_sampling
        self.centroid_sampling_weight = centroid_sampling_weight
        self.files = np.array(files)
        self.reader = FileReaderManager()
        self.num_points = np.zeros(len(self.files))
        self.num_points_per_file_and_class = np.zeros((len(self.files), len(class_ids)))
        self.file_hashs = []

        for idx, file in enumerate(self.files):
            point_cloud = (self.reader.read(file)).data()
            self.num_points[idx] = len(point_cloud)
            labels = point_cloud['semclassid']
            for class_id in class_ids:
                self.num_points_per_file_and_class[idx, class_id] = (labels == class_id).sum()
            self.file_hashs.append(hash_dataset(point_cloud))

        self.total_points = int(np.sum(self.num_points))
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.save_trees = save_trees
        self.k = k
        self.neighborhood_type = neighborhood_type
        self.neighborhood_sampling = neighborhood_sampling
        self.sampling_grid_size = sampling_grid_size
        self.sampling_feature = sampling_feature
        self.stratified_sampling_strategy = stratified_sampling_strategy
        self.k_percentages = k_percentages
        self.load_files_async = load_files_async
        self.loss_weight_attribute = loss_weight_attribute
        self.on_load_new_file = on_load_new_file
        self.known_columns = known_columns
        self.seed = seed
        if n != -1:
            self.points_to_draw = n
        else:
            self.points_to_draw = get_length(self.total_points, self.k, length_divider)

        if self.centroid_sampling == 'class_weights':
            centroid_sampling_weight = self.centroid_sampling_weight
        else:
            centroid_sampling_weight = 'unweighted'
        self.multi_sampler = MultiPointCloudSampler(self.num_points_per_file_and_class, self.points_to_draw,
                                                    centroid_sampling_weight, batch_size=batch_size)
        self.shuffling_random_generator = np.random.default_rng(seed)
        self.shuffled_file_indices = self.shuffling_random_generator.permutation(len(self.files))

        self._progress_bar = None

        self.thread_pool = futures.ThreadPoolExecutor()
        self.next_dataset: Optional[futures.Future[PointCloudDataset]] = None
        self.loader: Any = None
        self.loader_next_counter = 0
        self.radius = radius
        self.file_counter = 0
        self.file_idx = self.shuffled_file_indices[self.file_counter]

    def get_progress_bar(self) -> tqdm:
        """
        Returns a progress bar showing the dataloading progress. If no progress bar exists, a new one is created.

        :return: Progress bar object
        :rtype: tqdm
        """
        if self._progress_bar is None:
            total = self.multi_sampler.total_batches()
            self._progress_bar = tqdm(total, ascii=True, leave=False, file=sys.stdout)
        return self._progress_bar

    def _create_sampler(self,
                        dataset: PointCloudDataset,
                        num_samples: int,
                        replacement: bool = False,
                        generator: Optional[torch.Generator] = None) -> Sampler:
        """
        Creates sampler for drawing points from a point cloud dataset.

        :param dataset: Point cloud dataset from which samples are to be drawn.
        :type dataset: PointCloudDataset
        :param num_samples: Number of samples to be drawn.
        :type num_samples: integer
        :param replacement: If `True`, samples are drawn with replacement. If not, they are drawn without replacement.
            Defaults to `False`.
        :type replacement: bool, optional
        :param generator: Random generator to be used for sampling. Defaults to `None`.
        :type generator: torch.Generator, optional
        :return: Sampler object.
        :rtype: torch.utils.data.Sampler
        """

        if self.centroid_sampling == 'class_weights':
            return WeightedRandomSampler(list(dataset.sampling_weights), num_samples, replacement=replacement,
                                         generator=generator)

        return RandomSampler(dataset, num_samples=num_samples, replacement=replacement, generator=generator)

    def _create_random_generators(self) -> Tuple[np.random.Generator, Optional[torch.Generator]]:
        """
        Creates new random generators seeded with `self.seed`.

        :return: Random generators for numpy and PyTorch.
        :rtype: Tuple[np.random.Generator, Optional[torch.Generator]]
        """

        numpy_random_generator = np.random.default_rng(self.seed)
        if self.seed is not None:
            torch_random_generator = torch.Generator()
            torch_random_generator.manual_seed(self.seed)
        else:
            torch_random_generator = None

        return numpy_random_generator, torch_random_generator

    def _create_next_dataset(self) -> PointCloudDataset:
        """
        Creates dataset for the next point cloud file to be processed.

        :return: Dataset for the next point cloud file.
        :rtype: PointCloudDataset
        """

        return PointCloudDataset(
            self.files[self.file_idx],
            self.save_trees,
            self.k,
            self.neighborhood_type,
            self.neighborhood_sampling,
            self.radius,
            sampling_grid_size=self.sampling_grid_size,
            sampling_feature=self.sampling_feature,
            stratified_sampling_strategy=self.stratified_sampling_strategy,
            k_percentages=self.k_percentages,
            cache_dir=self.cache_dir,
            known_columns=self.known_columns,
            loss_weight_attribute=self.loss_weight_attribute,
            sampling_weights=self.multi_sampler.per_point_sampling_weights(self.file_idx),
            transform=self.transform
        )

    def _create_dataloader(self,
                           dataset: PointCloudDataset,
                           random_generator: Optional[torch.Generator] = None) -> DataLoader:
        """
        Creates a data loader for a given dataset.

        :param dataset: Dataset for which to create a data loader.
        :type dataset: PointCloudDataset
        :param random_generator: Random generator to be used for data sampling and shuffling.
        :type random_generator: torch.Generator, optional
        :return: Data loader
        :rtype: torch.utils.data.DataLoader
        """

        num_samples = self.multi_sampler.num_samples_per_file(self.file_idx)
        sampler = self._create_sampler(dataset, num_samples, replacement=False, generator=random_generator)

        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=random_generator
        )
        return loader

    def _load_next_file(self) -> None:
        """ Load new file and calculate number of samples (neighborhoods) to take from this file
        in order to create a corresponding DataLoader.
        """
        if self.file_counter == len(self.files):
            if self._progress_bar is not None:
                self._progress_bar.close()
            self._progress_bar = None
            raise StopIteration()
        del self.loader  # Reset loader before loading new files

        num_samples = self.multi_sampler.num_samples_per_file(self.file_idx)

        # continue with next file if no files are to be drawn from the current file
        if num_samples == 0:
            self._load_next_file()

        if self.next_dataset is None:
            self.next_dataset = self.thread_pool.submit(self._create_next_dataset)

        dataset = self.next_dataset.result()

        if self.on_load_new_file is not None:
            # a separate dataloader is created for the callback so that the random state of the ordinary data loading is
            # not changed by the callback
            numpy_random_generator, torch_random_generator = self._create_random_generators()
            dataset.random_generator = numpy_random_generator
            dataloader = self._create_dataloader(dataset, random_generator=torch_random_generator)

            self.on_load_new_file(dataloader, dataset.id())

        numpy_random_generator, torch_random_generator = self._create_random_generators()
        dataset.random_generator = numpy_random_generator
        dataloader = self._create_dataloader(dataset, random_generator=torch_random_generator)
        self.loader = iter(dataloader)

        self.file_counter += 1
        if self.file_counter < len(self.files):
            self.file_idx = self.shuffled_file_indices[self.file_counter]

        if self.load_files_async and self.file_counter != len(self.files):
            # load the next point cloud file in a background thread while iteration over the currently loaded dataset
            self.next_dataset = self.thread_pool.submit(self._create_next_dataset)
        else:
            self.next_dataset = None

    def id(self) -> str:
        """
        :return: ID of the dataloader.
        :rtype: string.
        """

        id_parts = []

        for file_hash, file_path in zip(self.file_hashs, self.files):
            id_parts.append(file_hash)
            id_parts.append(str(Path(file_path).stem))

        return '_'.join(id_parts)

    def __iter__(self) -> 'MultiPointCloudLoader':
        """ Randomly arranges files for diverse training epochs and returns an iterable object.
        """
        if self._progress_bar is not None:
            total = self.multi_sampler.total_batches()
            self._progress_bar.reset(total=total)
        self.shuffling_random_generator = np.random.default_rng(self.seed)
        self.shuffled_file_indices = self.shuffling_random_generator.permutation(len(self.files))
        self.file_counter = 0
        self.file_idx = self.shuffled_file_indices[self.file_counter]
        self.next_dataset = None
        self._load_next_file()
        return self

    def __next__(self) -> List[torch.Tensor]:
        """Keeps track of number of next() calls on the current DataLoader.
        If the current :code:`DataLoader` was called as for as many samples as intended for the corresponding file,
        the next file will be used.

        :return: batch of 'self.batch_size'-many neighborhoods, each containing k points
        :rtype: torch.tensor
        """
        try:
            next_sample = next(self.loader)
        except StopIteration:
            self._load_next_file()
            next_sample = self.__next__()

        if self._progress_bar is not None:
            self._progress_bar.update()

        return next_sample

    def __len__(self) -> int:
        return self.multi_sampler.total_batches()


def get_dataloaders(paths_train: List[Path],
                    paths_test: List[Path],
                    cache_dir: str,
                    class_ids: List[int],
                    batch_size: int,
                    k: int,
                    neighborhood_type: NeighborhoodType,
                    neighborhood_sampling: NeighborhoodSampling,
                    radius: float,
                    length_divider: int,
                    create_calibration_train_loader: bool = False,
                    centroid_sampling: SamplingStrategy = 'class_weights',
                    centroid_sampling_weight: Literal['unweighted', 'square_root', 'balanced'] = 'square_root',
                    sampling_grid_size: Optional[float] = None,
                    sampling_feature: Optional[str] = None,
                    stratified_sampling_strategy: Optional[str] = None,
                    k_percentages: Optional[List[float]] = None,
                    num_samples: int = -1,
                    num_workers: int = 0,
                    known_columns: Optional[List[str]] = None,
                    load_files_async: bool = False,
                    loss_weight_attribute: Optional[str] = None,
                    seed: Optional[int] = None) -> Tuple[MultiPointCloudLoader, MultiPointCloudLoader,
                                                         Optional[MultiPointCloudLoader]]:
    """Creation method for both the training and testing versions of the :code:`MultiPointCloudLoader`.

    :param paths_train: list of training filenames as strings, e.g. ["data/train_file1.h5", "data/train_file2.h5"]
    :type paths_train: list
    :param paths_test: list of testing filenames as strings, e.g. ["data/test_file1.h5", "data/test_file2.h5"]
    :type paths_test: list
    :param cache_dir: Path to folder to be used for caching files.
    :type cache_dir: string
    :param class_ids: List of all class IDs used in the point cloud datasets.
    :type class_ids: List[integer]
    :param batch_size: size of each batch
    :type batch_size: integer
    :param k: number of points drawn for each neighborhood
    :type k: integer
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
    :param create_calibration_loader: Whether a separate data loader for model calibration should be created for the
        training data. Defaults to `False`.
    :type create_calibration_loader: bool, optional
    :param centroid_sampling: To obtain inputs for deep learning models, small-scale point clouds are sampled
        from the large-scale input point clouds. Each small-scale point cloud is obtained by selecting a centroid point
        from the large-scale point cloud and retrieving a certain point neighborhood of the centroid (e.g., a spherical
        neighborhood). The `sampling_strategy` parameter controls how centroids are sampled. The available options are:
        `'class_weights'` | `'uniform'`.

        - `'class_weights'`: Sampling of points is weighted by the inverse class frequencies.
        - `'uniform'`: All points have the same probability of being sampled.
    :type centroid_sampling: string, optional
    :param centroid_sampling_weight: How classes should be weighted in sampling: `'unweighted'` | `'square_root'` |
        `'balanced'`.

        - `'unweighted'`: Points are drawn randomly without considering class labels.
        - `'square_root'`: Points are assigned sampling weights based on their class label. The sampling weights are
          chosen so that the class frequencies of the sampled points are (approximately) proportional to the square
          root of the class frequencies before sampling.
        - `'balanced'`: Points are assigned sampling weights based on their class label. The sampling weights are
          chosen so that (approximately) the same number of points is drawn from each class.
          Only used if `sampling_strategy` is set to `'class_weights'`. Defaults to `'square_root'`.
    :type centroid_sampling_weight: string, optional
    :param sampling_grid_size: Minimum grid size for `grid_sampling`.
    :type sampling_grid_size: float, optional
    :param sampling_feature: Feature to use for `feature_based_sampling`, if applicable.
    :type sampling_feature: string, optional
    :param stratified_strategy: Strategy to use for the `stratified_sampling` approach.
    :type stratified_strategy: string, optional
    :param k_percentages: String containing the percentages of k to be assigned to each shell. Only used for the
        `stratified_sampling` strategy.
    :type k_percentages: List[float], optional
    :param radius: radius for the neighborhood queries in meters
    :type radius: float
    :param length_divider: determines the amount of points sampled from the data loaders. Higher divider means fewer
        points.
    :type length_divider: integer
    :param num_samples: Directly sets the amount of points sampled from the datasets in total. If the value is different
        than -1, this parameter will override the `length-divider` one. If it equals -1, `length_divider` will be
        used to determine the number of points to sample.
    :type num_samples: integer
    :param num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main
        process. Number is forwarded to PyTorch DataLoader. Process creation time might lead to bad performance when
        using high number of workers.
    :type num_workers: num_workers
    :param known_columns: Columns to be used as model inputs. Defaults to `None`, which means that all columns are used
        as model inputs.
    :type known_columns: List[str].
    :param loss_weight_attribute: Name of a point attribute that should be used to weight the loss of each point.
        Defaults to `None`.
    :type loss_weight_attribute: string, optional
    :param load_files_async: Whether the next point cloud file should be loaded asynchronously while iterating over the
        current one. Enabling this option improves loading speed but increases the dataloaders memory requirements.
        Defaults to `False`.
    :type load_files_async: bool, optional
    :param seed: Random seed to be used for seeding data shuffling. Defaults to `None`, which means that data shuffling
        is not seeded.
    :type seed: integer, optional
    :return: Three MultiPointCloudLoaders, one for training, one for testing, and one for model calibration on the
        training data. If `create_calibration_train_loader` is set to `False`, `None` is returned instead of the third
        MultiPointCloudLoader.
    :rtype: tuple of types (MultiPointCloudLoader, MultiPointCloudLoader, Optional[MultiPointCloudLoader])
    """
    train_loader = MultiPointCloudLoader(
        paths_train,
        cache_dir=cache_dir,
        class_ids=class_ids,
        save_trees=True,
        batch_size=batch_size,
        k=k,
        neighborhood_type=neighborhood_type,
        neighborhood_sampling=neighborhood_sampling,
        sampling_grid_size=sampling_grid_size,
        sampling_feature=sampling_feature,
        stratified_sampling_strategy=stratified_sampling_strategy,
        k_percentages=k_percentages,
        radius=radius,
        num_workers=num_workers,
        length_divider=length_divider,
        transform=True,
        n=num_samples,
        known_columns=known_columns,
        load_files_async=load_files_async,
        loss_weight_attribute=loss_weight_attribute,
        centroid_sampling=centroid_sampling,
        centroid_sampling_weight=centroid_sampling_weight,
        seed=seed
    )

    calibration_train_loader = None
    if create_calibration_train_loader:
        num_calibration_samples = int(math.ceil(num_samples * 0.2)) if num_samples > 0 else -1

        calibration_train_loader = MultiPointCloudLoader(
            paths_train,
            cache_dir=cache_dir,
            class_ids=class_ids,
            save_trees=True,
            batch_size=batch_size,
            k=k,
            neighborhood_type=neighborhood_type,
            neighborhood_sampling=neighborhood_sampling,
            sampling_grid_size=sampling_grid_size,
            sampling_feature=sampling_feature,
            stratified_sampling_strategy=stratified_sampling_strategy,
            k_percentages=k_percentages,
            radius=radius,
            num_workers=num_workers,
            length_divider=length_divider * 5,
            transform=False,
            n=num_calibration_samples,
            known_columns=known_columns,
            load_files_async=load_files_async,
            loss_weight_attribute=loss_weight_attribute,
            centroid_sampling='uniform',
            centroid_sampling_weight='unweighted',
            seed=seed
        )

    test_loader = MultiPointCloudLoader(
        paths_test,
        cache_dir=cache_dir,
        class_ids=class_ids,
        save_trees=True,
        batch_size=batch_size,
        k=k,
        neighborhood_type=neighborhood_type,
        neighborhood_sampling=neighborhood_sampling,
        sampling_grid_size=sampling_grid_size,
        sampling_feature=sampling_feature,
        stratified_sampling_strategy=stratified_sampling_strategy,
        k_percentages=k_percentages,
        radius=radius,
        num_workers=num_workers,
        length_divider=length_divider,
        transform=False,
        n=num_samples,
        known_columns=known_columns,
        load_files_async=load_files_async,
        loss_weight_attribute=loss_weight_attribute,
        centroid_sampling='uniform',
        centroid_sampling_weight='unweighted',
        seed=seed
    )
    return train_loader, test_loader, calibration_train_loader
