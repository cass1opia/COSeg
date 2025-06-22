"""Module providing I/O utility functions meant to be used throughout the project as needed."""

__all__ = ['adjust_header', 'copy_prediction', 'copy_prediction_between_files', 'ensure_xyz_ordering',
           'generate_output_folders', 'get_files_from_path', 'find_closest_counterpart', 'rename_normal_columns',
           'remap_normalized_values']

import glob
import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd


def adjust_header(dataset: pd.DataFrame) -> pd.DataFrame:
    """Converts header to all lower-case, renames label to semclassid, potentially renames normal columns
    and ensures that columns x y z are in that order.

    :param dataset: point cloud
    :type dataset: pandas.DataFrame
    :return: point cloud with adjusted header
    :rtype: pandas.DataFrame
    """
    dataset.columns = pd.Index([str(x).lower() for x in dataset.columns])
    if 'label' in dataset.columns:
        dataset.rename(columns={'label': 'semclassid'}, inplace=True)
    dataset = rename_normal_columns(dataset)
    dataset = ensure_xyz_ordering(dataset)
    return dataset


def rename_normal_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Renames normal column names from normals12x/y/z to nx/y/z.

    :param dataset: point cloud
    :type dataset: pandas.DataFrame
    :return: point cloud with normal columns renamed
    :rtype: pandas.DataFrame
    """
    if 'normals12x' in dataset.columns:
        dataset.rename(columns={'normals12x': 'nx', 'normals12y': 'ny', 'normals12z': 'nz'}, inplace=True)
    return dataset


def ensure_xyz_ordering(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Ensures that a point cloud always has the columns 'x', 'y' and 'z' in that order.

    :param dataset: point cloud with x,y,z columns
    :type dataset: pandas.DataFrame
    :return: point cloud with reordered columns
    :rtype: pandas.DataFrame
    """
    order: List[Any] = ['x', 'y', 'z']
    order.extend([c for c in dataset.columns if c not in order])
    return dataset[order]


def remap_normalized_values(dataset: pd.DataFrame,
                            min_values: Dict[str, Union[int, float]],
                            max_values: Dict[str, Union[int, float]],
                            data_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Reads the given metadata and remaps the normalized (0-1) values (e.g. intensity) to their original range using
    the min/max values saved in the metadata.

    :param dataset: point cloud
    :type dataset: pandas.DataFrame
    :param min_values: minimum values of columns before processing point cloud
    :type min_values: dict
    :param max_values: maximum values of columns before processing point cloud
    :type max_values: dict
    :param data_types: data types of columns before processing point cloud
    :type data_types: dict, optional
    :return: point cloud without any normalized values
    :rtype: pandas.DataFrame
    """
    assert min_values.keys() == max_values.keys(), (
        "min and max_value keys do not match. Make sure your metadata is well formed.")
    if data_types:
        assert min_values.keys() == data_types.keys(), "Data types must match the keys of min and max values."

    for column in min_values:
        if column in dataset:
            min_value = min_values[column]
            max_value = max_values[column]
            if data_types:
                dtype = np.dtype(data_types[column]).type
                dataset[column] = dataset[column].map(lambda val: dtype(val * (max_value - min_value) + min_value))
            else:
                dataset[column] = dataset[column].map(lambda val: round(val * (max_value - min_value) + min_value))

    return dataset


def get_files_from_path(path: Union[Path, str],
                        supported_file_endings: List[str],
                        recursive: bool,
                        file_list_name: str = "") -> List[Path]:
    """Extracts a list of file paths to files to be preprocessed from a path.

    :param path: path to file or folder containing the files to be read
    :type path: pathlib.Path or str
    :param supported_file_endings: supported file extensions
    :type supported_file_endings: list
    :param recursive: whether the folder is to be searched recursively
    :type recursive: bool
    :param file_list_name: used for logging in case of error, defaults to \"\"
    :type file_list_name: string, optional
    :return: paths to files of supported type to be preprocessed
    :rtype: list(pathlib.Path)
    """
    files: List[Union[Path, str]] = []
    logging.debug('Process all {} items, recursive={}'.format(', '.join(supported_file_endings), recursive))
    path = Path(path).resolve()
    if path.is_file():
        if path.suffix[1:] in supported_file_endings:
            files = [path]
    elif recursive:
        for file_ending in supported_file_endings:
            files.extend(glob.glob(str(path) + '/**/*.' + file_ending, recursive=True))
    else:
        for file_ending in supported_file_endings:
            files.extend(glob.glob(str(path) + '/*.' + file_ending))

    if len(files) == 0:
        err_code = "E700"
        logging.error("(Error {}) List of {} files is empty, check if your path is correct".format(err_code,
                                                                                                   file_list_name))
        exit(err_code)
    logging.debug("List of {} files: {}".format(file_list_name, files))
    return [Path(file) for file in files]


def generate_output_folders(source: Union[str, pathlib.Path], destination: Union[str, pathlib.Path, None],
                            input_files: list):
    """ Reflects directory structure in output, and recursively saves files in directory.
        This function creates empty directory replicating input directory, so the processed
        files can be saved in respective folders, such as 'train', 'test', and 'predict'.

    :param source: paths to files of supported type to be processed
    :type source: str, pathlib.Path
    :param destination: path to folder for storing processed files
    :type destination: str, pathlib.Path, None
    :param input_files: list of paths to files to be processed
    :type input_files: list(pathlib.Path)
    :return: output_folders
    :rtype: list(pathlib.Path)
    """
    if destination is None:
        return None

    # getting the absolute path of the source directory
    source = pathlib.Path(source).resolve()
    if source.is_file():
        source = source.parent
    destination = pathlib.Path(destination).resolve()

    # creating paths to the destination folder
    output_folders = []
    for file in input_files:
        file = pathlib.Path(file).resolve()
        root_in_destination = destination / file.relative_to(source)
        output_folders.append(root_in_destination.parent)

    # Create subdirectories in destination folder
    for path in output_folders:
        if not Path(path).exists():
            os.makedirs(path)
    return output_folders


def find_closest_counterpart(files: Iterable[Path], files_to_compare: Iterable[Path]) -> Dict[Path, Optional[Path]]:
    original_to_output: Dict[Path, Optional[Path]] = {}
    for file in files:
        file_name = file.stem
        difference = float('inf')
        file_to_return = None
        for file_to_compare in files_to_compare:
            file_name_to_compare = file_to_compare.stem
            if file_name == file_name_to_compare:
                original_to_output[file] = file_to_compare
            if file_name in file_name_to_compare:
                new_difference = len(file_name_to_compare) - len(file_name)
                if new_difference < difference:
                    difference = new_difference
                    file_to_return = file_to_compare
        original_to_output[file] = file_to_return
    return original_to_output


def clean_up_files(files: Iterable[Path]) -> None:
    for file in files:
        file.unlink()


def copy_prediction(from_file: Union[str, Path], to_file: Union[str, Path], reader, writer):
    logging.info(f"Copy prediction from {from_file} to {to_file}")
    to_data = reader.read(to_file)
    from_data = reader.read(from_file, columns=['semclassidpredicted', 'specificclassidpredicted'])
    to_df = to_data.data()
    from_df = from_data.data()
    to_df['semclassidpredicted'] = from_df['semclassidpredicted']
    to_df['specificclassidpredicted'] = from_df['specificclassidpredicted']

    to_data.update_data(to_df)
    writer.write(to_data, to_file)


def copy_prediction_between_files(file_matching: Dict[str, Union[str, Path]]):
    from .file_handler import FileReaderManager, FileWriterManager
    reader = FileReaderManager()
    writer = FileWriterManager()
    for input_file, predicted_file in file_matching.items():
        copy_prediction(predicted_file, input_file, reader, writer)
