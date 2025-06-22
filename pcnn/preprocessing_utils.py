"""Provides utility functions used only during the preprocessing of point clouds for training and prediction."""

__all__ = ['access_temporary_file', 'get_closest_to_centroid', 'generate_semclass_dicts', 'get_temporary_path',
           'grid_subsampling_np', 'grid_subsampling_pd', 'save_processed_point_cloud', 'save_temporary_file',
           'simple_split_pc', 'tiled_grid_subsampling']

import hashlib
import json
import logging
from pathlib import Path
import psutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import globals

from .point_cloud import PointCloud
from .file_handler import FileReaderManager, FileWriterManager
from .general_utils import pair_function


def _get_unique_pair(baseclassid: int,
                     base_class_name: str,
                     semclassid: int,
                     semclassname: str) -> Dict[Tuple[str, str], int]:
    """Creates a dictionary to map the baseclass and specificclass to the unique_id.

    :param baseclassid: id of the base class
    :type baseclassid: unsigned integer
    :param base_class_name: name of the base class
    :type base_class_name: string
    :param semclassid: id of the specific class
    :type semclassid: unsigned integer
    :param semclassname: name of specific class
    :type semclassname: string
    :return: dictionary to map the base class name and specific class name to their unique_id
    :rtype: dicitionary
    """
    return {(base_class_name, semclassname): pair_function(baseclassid, semclassid)}


def generate_semclass_dicts(json_path: str) -> Tuple[Dict[Tuple[str, str], int],
                                                     Dict[Tuple[str, str], Tuple[int, int]]]:
    """Creates two complete dictionaries, the first contains the mapping from base and specific class name to the
    unique_id and the second contains the mapping from base and specific class name to their normal id.

    :param json_path: Path to the .json-file, that contains the information about the different specific classes
    :type json_path: string
    :return: the dictionary (class_list), containing the mapping from base and specific class name to the unique id,
        and the dictionary (string_to_id), containing the mapping from class name to class id
    :rytpe: tuple of two dictionaries
    """
    with open(json_path, 'r') as f:
        datastore = json.load(f)
    class_list: Dict[Tuple[str, str], int] = {}
    string_to_id: Dict[Tuple[str, str], Tuple[int, int]] = {}
    bc_map = datastore['base_class_map']
    hierarchy = datastore['hierarchy']
    for id in bc_map:
        id_int = int(id)
        class_list.update(_get_unique_pair(id_int, bc_map[id], 0, 'None'))
        string_to_id.update({(bc_map[id], 'None'): (id_int, 0)})
    for b_id in hierarchy:
        b_id_int = int(b_id)
        b_name = bc_map[b_id]
        for spec_id in hierarchy[b_id]:
            spec_id_int = int(spec_id)
            class_list.update(_get_unique_pair(b_id_int, b_name, spec_id_int, hierarchy[b_id][spec_id]))
            string_to_id.update({(b_name, hierarchy[b_id][spec_id]): (b_id_int, spec_id_int)})
    return class_list, string_to_id


def _save_split(point_cloud: PointCloud, writer: FileWriterManager, idx_low: int, idx_high: int,
                out_name: Path) -> None:
    point_cloud = PointCloud(point_cloud.data().iloc[idx_low: idx_high])
    writer.write(point_cloud, out_name)


def simple_split_pc(point_cloud: PointCloud, out_path: Path, max_points: int, overlap: int) -> None:
    """Attempts to split the point cloud into segments based of index and saves the segments. Includes overlap between
    segments.

    :param point_cloud: point cloud
    :type point_cloud: PointCloud
    :param out_path: output path for the point cloud
    :type out_path: pathlib.Path
    :param max_points: maximum number of points that a part of the point cloud can have
    :type max_points: int
    :param overlap: number of points that are overlap between two segments
    :type overlap: int
    """
    writer = FileWriterManager()
    idx_low = np.array(range(0, len(point_cloud), max_points - overlap))
    idx_high = idx_low + overlap

    for i in range(len(idx_low) - 1):
        new_path = out_path.parent / (out_path.stem + f'_part{i}' + '.h5')
        _save_split(point_cloud, writer, idx_low[i], idx_high[i + 1], new_path)
    new_path = out_path.parent / (out_path.stem + f'_part{len(idx_low)}' + '.h5')
    _save_split(point_cloud, writer, idx_low[-1], len(point_cloud), new_path)


def save_processed_point_cloud(path_to_file: Path, point_cloud: PointCloud, split: bool,
                               output_folder: Optional[Path] = None):
    """Saves the preprocessed point cloud as #name#_preprocessed.h5 in either the specified output folder or the input
    file's directory.

    :param path_to_file: path to on of the input files
    :type path_to_file: pathlib.Path
    :param point_cloud: point cloud
    :type point_cloud: PointCloud
    :param split: determines if the point cloud should be split if it is too large
    :type split: bool
    :param output_folder: path to folder for preprocessed files, defaults to None
    :type output_folder: pathlib.Path, optional
    """
    # the training only allows 2^24 points, which is a little over 16 million. The values are chosen arbitrarily such
    # that split point clouds still work for training.
    SPLIT_THRESHOLD = 16 * 10 ** 6
    OVERLAP = 5 * 10 ** 5
    if output_folder is None:
        output_folder = path_to_file.parent
    else:
        output_folder = output_folder
    out_name = Path(output_folder, path_to_file.with_name(path_to_file.stem + '_preprocessed' + '.h5').name)
    if len(point_cloud) > SPLIT_THRESHOLD:
        if split:
            simple_split_pc(point_cloud, out_name, SPLIT_THRESHOLD, OVERLAP)
        else:
            logging.warning(f"Point cloud {path_to_file.name} has more than 16 million points(has {len(point_cloud)}."
                            f"This means it cannot be used for training. Consider using --split.")
            writer = FileWriterManager()
            writer.write(point_cloud, out_name)
    else:
        writer = FileWriterManager()
        writer.write(point_cloud, out_name)


def get_temporary_path(path_to_file: Path, settings: Dict[str, Any]) -> Path:
    """Generates a specific temporary path for saving files during preprocessing. The path depends on the input file
    path.

    :param path_to_file: path to the input file, the dataset was read from
    :type path_to_file: pathlib.Path
    :param settings: global settings object
    :type settings: dict
    :return: temporary file path for the input file
    :rtype: pathlib.Path
    """
    return Path(settings['temporary_folder'], str(hashlib.md5(str(path_to_file).encode('utf-8')).hexdigest()) + '.h5')


def save_temporary_file(point_cloud: PointCloud, path_to_file: Path, settings: Dict[str, Any]) -> None:
    """Writes the data contained in the given dataset to an HDF5 file in a temporary directory.

    :param point_cloud: point cloud
    :type point_cloud: PointCloud
    :param path_to_file: path to the input file, the dataset was read from
    :type path_to_file: pathlib.Path
    :param settings: global settings object
    :type settings: dict
    """
    temp_path = get_temporary_path(path_to_file, settings)
    writer = FileWriterManager()
    writer.write(point_cloud, temp_path)


def access_temporary_file(path_to_file: Path, settings: Dict[str, Any]) -> PointCloud:
    """Reads a temporary point cloud file generated during metadata calculation from its temporary location and deletes
    it afterwards.

    :param path_to_file: path to the temporary file
    :type path_to_file: pathlib.Path
    :param settings: global settings object
    :type settings: dict
    :return: point cloud, read from temporary file
    :rtype: PointCloud
    """
    temp_path = get_temporary_path(path_to_file, settings)
    reader = FileReaderManager()
    point_cloud = reader.read(temp_path)
    temp_path.unlink()  # delete the file
    return point_cloud


def get_closest_to_centroid(data: np.ndarray, index: np.ndarray) -> int:
    """Takes a voxel of points and returns the point closest to the voxel\'s centroid.

    :param data: point cloud data
    :type data: np.ndarray
    :param index: indices of the points to be considered for calculation
    :type index: np.array
    :return: index of the point closest to the voxel\'s centroid
    :rtype: int
    """
    if len(index) == 1:
        return index[0]
    return index[np.argmin(np.linalg.norm(data[index][:, :3] - data[index][:, :3].mean(axis=0), axis=1))]


def tile_point_cloud(point_cloud: pd.DataFrame, tile_size: np.ndarray) -> List[pd.DataFrame]:
    """
    Divides a point cloud into a set of tiles.

    :param point_cloud: Point cloud to be tiled. Must contain at least contain the columns `x`, `y`, `z`.
    :type point_cloud: pandas.DataFrame
    :param tile_size: Dimensions of the tiles along each axis. Must have shape `(3)`.
    :type tile_size: numpy.ndarray
    :return: List of point cloud tiles.
    :rtype: List[pandas.DataFrame]
    """

    coords = point_cloud[['x', 'y', 'z']].values

    max_coords = coords.max(axis=0)
    min_coords = np.floor(coords.min(axis=0))

    num_tiles = np.ceil((max_coords - min_coords) / tile_size)
    max_x = min_coords[0] + num_tiles[0] * tile_size[0]
    max_y = min_coords[1] + num_tiles[1] * tile_size[1]
    max_z = min_coords[2] + num_tiles[2] * tile_size[2]

    tiles = []
    current_x = min_coords[0]
    while current_x <= max_x:
        current_y = min_coords[1]
        while current_y <= max_y:
            current_z = min_coords[2]
            while current_z <= max_z:
                tile_start = np.array([[current_x, current_y, current_z]])
                tile_mask = np.logical_and((coords >= tile_start).all(axis=-1),
                                           (coords < tile_start + tile_size).all(axis=-1))
                current_tile = point_cloud[tile_mask]
                if len(current_tile) > 0:
                    tiles.append(current_tile)
                current_z += tile_size[2]
            current_y += tile_size[1]
        current_x += tile_size[0]

    return tiles


def grid_subsampling_pd(dataset: pd.DataFrame, grid_size: float, fast_density: bool = False,
                        num_workers: Optional[int] = None,
                        return_sorted: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Uses the passed grid size to group all points of the input point cloud into voxels. The voxels are used to
    extract either the first point or the point closest to the centroid, reducing the density of the point cloud.
    Point order is maintained.

    :param dataset: Point cloud to be downsampled.
    :type dataset: pandas.DataFrame
    :param grid_size: Grid size. If set to zero, no density reduction is performed.
    :type grid_size: float
    :param fast_density: If `True`, an arbitrary point of a grid cell is returend instead of the centroid. Defaults to
        `False`.
    :type fast_density: bool, optional
    :param num_workers: When this method is called by multiple worker processes in parallel, the memory used for
        grid subsampling should be equally split between these worker processes. To allow such splitting of the
        memory, the number of worker processes can be passed. If `num_workers` is `None`, this method assumes that it
        can use all the available memory.
    :type num_workers: integer, optional
    :parm return_sorted: Whether the k nearest neighbors are to be sorted by distance. Defaults to `True`.
    :type return_sorted: bool, optional

    :return: Downsampled point cloud and indices of the selected points.
    :rtype: Tuple[pandas.DataFrame, numpy.ndarray]
    """

    _, selected_indices = grid_subsampling_np(dataset[['x', 'y', 'z']].to_numpy(), grid_size,
                                              fast_density=fast_density, num_workers=num_workers,
                                              return_sorted=return_sorted)

    return dataset.iloc[selected_indices], selected_indices


def grid_subsampling_np(dataset: np.ndarray, grid_size: float, fast_density: bool = False,
                        num_workers: Optional[int] = None,
                        return_sorted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Uses the passed grid size to group all points of the input point cloud into voxels. The voxels are used to
    extract either the first point or the point closest to the centroid, reducing the density of the point cloud.
    Point order is maintained.

    :param dataset: Point cloud to be downsampled. Must have shape `(N, 3 + D)`, where
        `N = number of points before subsampling` and `D = number of feature channels`. The first three channels per
        point are expected to contain the point coordinates.
    :type dataset: numpy.ndarray
    :param grid_size: Grid size. If set to zero, no density reduction is performed.
    :type grid_size: float
    :param fast_density: If `True`, an arbitrary point of a grid cell is returend instead of the centroid. Defaults to
        `False`.
    :type fast_density: bool, optional
    :param num_workers: When this method is called by multiple worker processes in parallel, the memory used for
        grid subsampling should be equally split between these worker processes. To allow such splitting of the
        memory, the number of worker processes can be passed. If `num_workers` is `None`, this method assumes that it
        can use all the available memory.
    :type num_workers: integer, optional
    :parm return_sorted: Whether the k nearest neighbors are to be sorted by distance. Defaults to `True`.
    :type return_sorted: bool, optional

    :return: Downsampled point cloud and indices of the selected points.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    if grid_size == 0:
        return dataset, np.arange(len(dataset), dtype=np.int64)

    coords = dataset[:, :3]

    if fast_density:
        to_bin = np.floor(coords / grid_size).astype(np.int64)
        _, to_bin = np.unique(to_bin, axis=0, return_index=True)
        if return_sorted:
            to_bin = np.sort(to_bin, kind='heapsort')

        return dataset[to_bin], to_bin
    else:
        # in this (slower) grid sampling implementation, for each bin, the points falling into that bin are
        # collected and the point closest to the centroid of the points within a bin is selected
        # if GPU is available this grid subsampling algorithm is executed on GPU
        # if the available memory is too small to process the point cloud as a whole it is split into chunks

        device = globals.torch_device

        # determine available memory
        if globals.cuda_available:
            # available GPU memory
            available_memory = torch.cuda.mem_get_info(device=device)[0]  # type: ignore [index]
        else:
            # available CPU memory
            available_memory = int(psutil.virtual_memory().available)

        reserve = 0.5  # reserve 50 % of the available memory for other applications
        max_memory_to_use = 4 * 1e+09  # use 4 GB chunks at maximum
        available_memory = min(available_memory * reserve, max_memory_to_use)

        # if there are parallel worker processes, the memory has to be split between the workers
        if num_workers is not None:
            available_memory = available_memory // num_workers

        # process point cloud in tiles that fit into the available memory
        selected_indices = tiled_grid_subsampling(coords, np.arange(len(dataset)), grid_size,
                                                  available_memory, device)

        return dataset[selected_indices], selected_indices


def tiled_grid_subsampling(point_coords: np.ndarray, point_idx: np.ndarray, grid_size: float, available_memory: int,
                           device: torch.device, return_sorted: bool = True) -> np.ndarray:
    """
    Subsamples the point cloud using grid subsampling. From each grid cell, the point closest to the centroid is
    returend. If the available memory is to small to process the point cloud as a whole, an octree-like approach is used
    to recursively split the point cloud into tiles until the tiles are small enough to be processed. The chosen tile
    sizes are always multiples of the specified grid size.

    :param point_coords: Point cloud to be downsampled. Must have shape `(N, 3)`, where `N = number of points`.
    :type point_coords: numpy.ndarray
    :param point_idx: Point indices. Must have shape `(N)`, where `N = number of points`.
    :type point_idx: numpy.ndarray
    :param grid_size: Grid size.
    :type grid_size: float
    :param available_memory: Amount of memory that is available for processing (in bytes).
    :type available_memory: int
    :param device: PyTorch device on which is the downsampling is to be executed.
    :type device: torch.device
    :parm return_sorted: Whether the k nearest neighbors are to be sorted by distance. Defaults to `True`.
    :type return_sorted: bool, optional

    :return: Indices of the selected points.
    :rytpe: numpy.ndarray
    """

    num_points = len(point_coords)
    float_size = torch.tensor([], dtype=torch.float).element_size()  # element size of float tensors in byte
    long_size = torch.tensor([], dtype=torch.long).element_size()  # element size of long tensors in byte

    # here, a rough estimate of the memory needed to process the point cloud as a whole is computed
    # for processing, at least the following tensors must fit into memory
    #   coords: num_points * size_float
    #   to_bin: num_points * size_long
    #   inv_indices: num_points * size_long
    #   points_counts_per_cell: num_points * size_long
    #   sorted_points_counts_per_cell: num_points * size_long (at maximum)
    #   point_indices: num_points * size_long
    #   inv_points_counts_per_cell: num_points * size_long
    #   sorted_idx: num_points * size_long
    # to ensure that some memory will be left for the other processing steps, the memory needed for the above tensors is
    # doubled (this is only a rough heuristic)
    memory_needed = num_points * (float_size + 7 * long_size) * 2

    if memory_needed > available_memory:
        # split point cloud into eight tiles / octants and process each tile separately
        eps = 1e-5  # add small epsilon to avoid that points are exactly on the tile border
        tile_size = (point_coords.max(axis=0) - np.floor(point_coords.min(axis=0)) + eps) / 2

        # ensure that tile size is a multiple of the grid size
        tile_size = np.ceil(tile_size / grid_size) * grid_size

        if (tile_size > grid_size).any():
            selected_indices_list = []
            point_cloud_df = pd.DataFrame(np.concatenate([point_coords, np.expand_dims(point_idx, -1)], axis=1),
                                          columns=['x', 'y', 'z', 'idx'])
            for tile in tile_point_cloud(point_cloud_df, tile_size):
                current_selected_indices = tiled_grid_subsampling(tile[['x', 'y', 'z']].values,
                                                                  tile['idx'].values.astype(np.int64),
                                                                  grid_size, available_memory, device)

                selected_indices_list.append(current_selected_indices)

            selected_indices = np.concatenate(selected_indices_list)
            selected_indices.sort()
            return selected_indices
        else:
            # all points are located in the same grid cell, so the point closest to the centroid can be returned
            centroid_index = point_idx[get_closest_to_centroid(point_coords, np.arange(len(point_coords)))]
            return np.expand_dims(centroid_index, -1)

    # the point cloud is small enough to be processed without tiling in this case

    # load point coords onto GPU if one is available
    coords = torch.tensor(point_coords, dtype=torch.float, device=device)
    point_indices = torch.arange(num_points, dtype=torch.long, device=device)
    del point_coords

    invalid_index = num_points

    # compute in which grid cell each point is located
    to_bin = torch.div(coords, grid_size, rounding_mode='floor').long()

    # for each grid cell, compute the indices of the contained points and the number of contained points
    _, inv_indices, points_counts_per_cell = torch.unique(to_bin, dim=0, return_inverse=True, return_counts=True)
    del to_bin

    sorted_points_counts_per_cell = torch.unique(points_counts_per_cell, sorted=True)

    # for each point, collect the number of points that are in the same bin
    inv_points_counts_per_cell = torch.gather(points_counts_per_cell, 0, inv_indices)

    # sort tensors by grid cell index so that the points of each grid cell are stored in a continuous range
    inv_indices, sorted_idx = torch.sort(inv_indices)
    point_indices = point_indices[sorted_idx]
    inv_points_counts_per_cell = inv_points_counts_per_cell[sorted_idx]
    del inv_indices
    del sorted_idx

    selected_indices_list = []

    # for each grid cell, we have to find the point closest to the centroid
    # for this purpose, the centroid of each grid cell has to be computed and the distance beteween each point in a grid
    # cell and the centroid
    # to do this computation in parallel on the GPU, the maximum number of points per grid cell is determined and a
    # tensor of shape (G, N_{max}) is created where `G = number of grid cells`, and `N_{max}` is the maximum number of
    # points per grid cell
    # for grid cells containing less than `N_{max}` points, the tensor is padded with dummy points
    # since the maximum number of points per grid cell usually is much larger than the average number of points per
    # grid cell, this padding may waste a lot of memory
    # to reduce memory consumption, the tensor is split into separate chunks where each chunk consists of those
    # grid cells that contain more points than a lower threshold and less points than an upper threshold
    # as each chunk consists of grid cells containing a similar number of points, this will reduce the padding and
    # thus the memory consumption

    # some of the memory is already filled with the tensors created before
    available_memory = available_memory - memory_needed

    max_points_per_cell = points_counts_per_cell.amax()

    # initialize chunk borders
    current_min_points_per_cell = 0
    current_max_points_per_cell = max_points_per_cell

    while current_min_points_per_cell < max_points_per_cell:
        # split chunk until it fits into available memory

        # first, try a chunk that covers all bins
        current_mask = torch.logical_and(points_counts_per_cell > current_min_points_per_cell,
                                         points_counts_per_cell <= current_max_points_per_cell)
        current_num_bins = current_mask.sum()
        # approximate amount of memory needed to process the chunk (only coarse approximation)
        # for each grid cell of the chunk, at least the indices and the coordinates of all contained must fit into
        # memory
        #   point_indices_per_bin: current_num_bins * current_max_points_per_cell * size_long
        #   points_per_cell: current_num_bins * current_max_points_per_cell * 3 * size_float
        #   centroids_per_cell: current_num_bins * 3 * size_float
        #   dists_to_centroids: current_num_bins * current_max_points_per_cell * size_float
        #   representative_indices: current_num_bins * size_long
        memory_needed = (current_num_bins * (current_max_points_per_cell * (long_size + 4 * float_size)
                                             + 3 * float_size
                                             + long_size))
        i = len(sorted_points_counts_per_cell) - 1
        while memory_needed > available_memory:
            if i == 0 or sorted_points_counts_per_cell[i] <= current_min_points_per_cell:
                break
            # split chunk into two parts
            current_max_points_per_cell = sorted_points_counts_per_cell[i]
            i -= 1
            current_mask = torch.logical_and(points_counts_per_cell > current_min_points_per_cell,
                                             points_counts_per_cell <= current_max_points_per_cell)
            current_num_bins = current_mask.sum()
            memory_needed = (current_num_bins * (current_max_points_per_cell * (long_size + 4 * float_size)
                                                 + 3 * float_size
                                                 + long_size))

        # select point indices of current chunk
        current_point_indices = point_indices[torch.logical_and(
            inv_points_counts_per_cell > current_min_points_per_cell,
            inv_points_counts_per_cell <= current_max_points_per_cell
        )]

        # create a fixed-shape tensor that contains the indices of the points falling into each grid cell of the chunk
        point_indices_per_bin = torch.arange(current_max_points_per_cell, device=device, dtype=torch.long).\
            unsqueeze(0).repeat(int(current_num_bins.item()), 1)

        point_indices_per_bin[
            point_indices_per_bin >= points_counts_per_cell[current_mask].unsqueeze(-1)] = invalid_index
        point_indices_per_bin = point_indices_per_bin.flatten()
        point_indices_per_bin[point_indices_per_bin != invalid_index] = current_point_indices
        point_indices_per_bin = point_indices_per_bin.reshape((-1, current_max_points_per_cell.item()))
        del current_point_indices

        # add dummy / shadow point that is returned for invalid indices
        # we set the dummy point to zero so it won't affect the computation of the centroid
        padded_coords = torch.concat([coords, torch.zeros((1, 3), device=device)])

        # gather coordinates of points within each bin
        padded_coords = padded_coords.unsqueeze(-2).expand(-1, current_max_points_per_cell, -1)
        points_per_cell = torch.gather(padded_coords, 0, point_indices_per_bin.unsqueeze(-1).expand(-1, -1, 3))
        del padded_coords

        # compute centroid of each bin and select point closest to that centroid as representative
        centroids_per_cell = points_per_cell.sum(dim=-2, keepdim=True) / points_counts_per_cell[
            current_mask].reshape(-1, 1, 1)
        del current_mask

        dists_to_centroids = torch.linalg.norm(points_per_cell - centroids_per_cell, dim=-1)
        dists_to_centroids[point_indices_per_bin == invalid_index] = torch.inf
        representative_indices = torch.argmin(dists_to_centroids, dim=-1, keepdim=True)
        del points_per_cell
        del centroids_per_cell
        del dists_to_centroids

        selected_indices = torch.gather(point_indices_per_bin, -1, representative_indices).flatten()

        selected_indices_list.append(selected_indices.cpu().numpy())
        del selected_indices

        # set borders of next chunk
        current_min_points_per_cell = current_max_points_per_cell
        current_max_points_per_cell = max_points_per_cell

    all_selected_indices = np.concatenate(selected_indices_list)
    all_selected_indices = point_idx[all_selected_indices]

    # sort indices to avoid that points are sorted by grid cell index
    if return_sorted:
        all_selected_indices.sort()

    return all_selected_indices
