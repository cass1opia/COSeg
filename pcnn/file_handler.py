__all__ = ['CSVFileHandler', 'CSVReader', 'CSVWriter', 'FileReader', 'FileReaderManager', 'FileWriter',
           'FileWriterManager', 'HDFFileHandler', 'HDFReader', 'HDFWriter', 'LASFileHandler', 'LASReader', 'LASWriter',
           'NoFileHandler', 'NoReader', 'NoWriter', 'XPCFileHandler', 'XPCReader', 'XPCWriter']

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import laspy
import numpy as np
import pandas as pd

from .point_cloud import PointCloud
from .io_utils import adjust_header


class FileReader(ABC):
    def read(self,
             path: Union[str, Path],
             columns: Optional[List[str]] = None,
             use_smallest_types: bool = True) -> PointCloud:
        """Handles PointCloud object creation and file system utility for the FileReaders before reading.

        :param path: path of the file to be read
        :type path: pathlib.Path or str
        :param columns: Point cloud columns / attributes to be read into memory. Defaults to `None`, which means that
            all columns are read.
        :type columns: List[str], optional
        :param use_smallest_datatypes: Whether for integer attributes the smallest possible data type should be used,
            with which all attribute values can be represented. Defaults to `True`.
        :type use_smallest_types: bool
        :return: created point cloud
        :rtype: PointCloud
        """
        df, shift = self.read_to_dataframe(path, columns)
        max_resolutions = self.read_max_resolutions(path)

        df = adjust_header(df)

        if use_smallest_types:
            df = self.reduce_memory_usage(df)

        return PointCloud(df, *shift, *max_resolutions)

    @abstractmethod
    def read_to_dataframe(self,
                          path: Union[str, Path],
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
        pass

    @staticmethod
    def read_max_resolutions(path: Union[str, Path]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Reads the maximum resolution for each coordinate dimension from the header of the point cloud file.

        :param path: Path of the file to be read.
        :type path: pathlib.Path or str
        :return: Maximum resolution of the x-, y-, and z-coordinates of the point cloud. If the file format does not
            support setting a maximum resolution, `None` values are returned instead.
        :rtype: Tuple[Optional[float], Optional[float], Optional[float]]
        """

        return (None, None, None)

    @staticmethod
    def supported_types() -> List[str]:
        return []

    @staticmethod
    def _get_columns_filter(columns: List[str]) -> List[str]:
        """
        Converts a list of column names into a list of columns that can be used to filter the columns to be loaded.
        Adds the "x", "y" and "z" columns as they always have to be loaded.

        :param columns: A list of column names to be used as column filter.
        :type columns: List[str]
        :return: Sanitized list of column names.
        :rtype: List[str]
        """

        columns.extend(["x", "y", "z"])
        columns_lower_case = []
        for column in columns:
            columns_lower_case.append(column.lower())
        columns_lower_case = list(set(columns_lower_case))
        return columns_lower_case

    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame):
        """
        Converts data types of integer attributes of a pandas DataFrame to the smallest possible type that can represent
        all attribute values.

        :param df: Data frame whose data types are to be converted.
        :type df: pandas.DataFrame
        :return: Data frame with converted data types.
        :rtype: pandas.DataFrame
        """

        # adapted code from
        # https://codereview.stackexchange.com/questions/168746/apply-the-smallest-possible-datatype-for-each-column-in
        # -a-pandas-dataframe-to-re

        for column in df.columns:
            if pd.api.types.is_integer_dtype(df[column]):
                min_value = df[column].min()
                max_value = df[column].max()

                if min_value > np.iinfo(np.int8).min and max_value < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.int8)
                elif min_value > np.iinfo(np.int16).min and max_value < np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.int16)
                elif min_value > np.iinfo(np.int32).min and max_value < np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.int32)
                elif min_value > np.iinfo(np.int64).min and max_value < np.iinfo(np.int64).max:
                    df[column] = df[column].astype(np.int64)

        return df


class FileWriter(ABC):
    def write(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        """Handles file system utility for the FileWriters before writing.

        :param point_cloud: point cloud to be written
        :type point_cloud: PointCloud
        :param path: path of the file to be written
        :type path: pathlib.Path or str
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write_to_file(point_cloud, path)

    @abstractmethod
    def write_to_file(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        pass

    @staticmethod
    def supported_types() -> List[str]:
        return []


class CSVFileHandler:
    @staticmethod
    def supported_types() -> List[str]:
        return ['.csv', '.txt']


class CSVReader(CSVFileHandler, FileReader):
    def __init__(self, separator: str = ',') -> None:
        self._separator = separator

    def read_to_dataframe(self,
                          path: Union[str, Path],
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
        """
        :param path: path of the file to be read
        :type path: pathlib.Path or str
        :param columns: Point cloud columns / attributes to be read into memory. Defaults to `None`, which means that
            all columns are read.
        :type columns: List[str], optional
        :return: point cloud data as dataframe and shift
        :rtype: tuple(pandas.DataFrame, tuple(int))
        """

        columns_filter = self._get_columns_filter(columns=columns) if columns is not None else None

        return pd.read_csv(path,
                           sep=self._separator,
                           usecols=lambda x: columns_filter is None or x.lower() in columns_filter
                           ), (0, 0, 0)


class CSVWriter(CSVFileHandler, FileWriter):
    def __init__(self, separator: str = ','):
        self._separator = separator

    def write_to_file(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        """
        :param point_cloud: point cloud to be written
        :type point_cloud: PointCloud
        :param path: path of the file to be written
        :type path: pathlib.Path or str
        """
        point_cloud.data_shifted().to_csv(path, sep=self._separator, index=False)


class HDFFileHandler:
    @staticmethod
    def supported_types() -> List[str]:
        return ['.h5', '.hdf']


class HDFReader(HDFFileHandler, FileReader):
    def read_to_dataframe(self,
                          path: Union[str, Path],
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
        """
        :param path: path of the file to be read
        :type path: pathlib.Path or str
        :param columns: Point cloud columns / attributes to be read into memory. Defaults to `None`, which means that
            all columns are read.
        :type columns: List[str], optional
        :return: point cloud data as dataframe and shift
        :rtype: tuple(pandas.DataFrame, tuple(int))
        """
        df = pd.DataFrame(pd.read_hdf(path, 'table'))
        if columns is not None:
            df.columns = pd.Index([str(x).lower() for x in df.columns])
            df = df[[column for column in df.columns if column in set(self._get_columns_filter(columns))]]

        shift = self._read_shift(path)
        return df, shift

    def _read_shift(self, path: Union[str, Path]) -> Tuple[int, int, int]:
        with pd.HDFStore(path) as store:
            try:
                # get_storer() accesses a pytables object
                x_shift = store.get_storer('table').attrs.x_shift  # type: ignore
                y_shift = store.get_storer('table').attrs.y_shift  # type: ignore
                z_shift = store.get_storer('table').attrs.z_shift  # type: ignore
                return x_shift, y_shift, z_shift
            except AttributeError:
                return 0, 0, 0


class HDFWriter(HDFFileHandler, FileWriter):
    def write_to_file(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        """
        :param point_cloud: point cloud to be written
        :type point_cloud: PointCloud
        :param path: path of the file to be written
        :type path: pathlib.Path or str
        """
        point_cloud.data().to_hdf(path, 'table', append=False, mode='w')
        with pd.HDFStore(path) as store:
            # get_storer() accesses a pytables object
            store.get_storer('table').attrs.x_shift = point_cloud.x_shift  # type: ignore
            store.get_storer('table').attrs.y_shift = point_cloud.y_shift  # type: ignore
            store.get_storer('table').attrs.z_shift = point_cloud.z_shift  # type: ignore


class XPCFileHandler:
    def __init__(self):
        self._xpc_to_supported_column_names = {
            "xyz_x": "x",
            "xyz_y": "y",
            "xyz_z": "z",
            "las_intensity": "intensity",
            "rgb_x": "r",
            "rgb_y": "g",
            "rgb_z": "b",
            "rgba_x": "r",
            "rgba_y": "g",
            "rgba_z": "b",
            "semantic_class": "semclassid",
            "normal_12_x": "nx",
            "normal_12_y": "ny",
            "normal_12_z": "nz",
            "normal_3_x": "normals3x",
            "normal_3_y": "normals3y",
            "normal_3_z": "normals3z",
            "curvature": "curvature",
            "specific_semantic_class": "specificclassid"
        }

    @staticmethod
    def supported_types() -> List[str]:
        return ['.xpc']


class XPCReader(XPCFileHandler, FileReader):
    def read_to_dataframe(self,
                          path: Union[str, Path],
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
        """
        :param path: path of the file to be read
        :type path: pathlib.Path or str
        :param columns: Point cloud columns / attributes to be read into memory. Defaults to `None`, which means that
            all columns are read.
        :type columns: List[str], optional
        :return: point cloud data as dataframe and shift
        :rtype: tuple(pandas.DataFrame, tuple(int))
        """
        try:
            import xpc_wrapper
        except ImportError as e:
            logging.error(e)
            return pd.DataFrame(), (0, 0, 0)
        # if path is not string, convert to string as the xpc_wrapper's Cpp interface needs that type
        if not isinstance(path, str):
            path = str(path.resolve())

        supported_attributes = self._xpc_to_supported_column_names.keys()
        supported_attributes = [column.lower() for column in supported_attributes]
        supported_attributes = list(set(supported_attributes).intersection(xpc_wrapper.get_available_attributes(path)))
        if columns is not None:
            supported_attributes = list(set(supported_attributes).intersection(set(self._get_columns_filter(columns))))
        python_wrapped_point_cloud = xpc_wrapper.load(path, supported_attributes)
        df = pd.DataFrame.from_dict(python_wrapped_point_cloud.attributes_map)
        df.rename(self._xpc_to_supported_column_names, axis='columns', inplace=True)
        return df, python_wrapped_point_cloud.bounding_box_shift


class XPCWriter(XPCFileHandler, FileWriter):
    def write_to_file(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        """Writes point cloud to xpc file. This function handles data conversion as well as renaming prediction specific
        columns amongst other things.

        :param point_cloud: point cloud to be written
        :type point_cloud: PointCloud
        :param path: path of the file to be written
        :type path: pathlib.Path or str
        """
        try:
            import xpc_wrapper
        except ImportError as e:
            logging.error(e)
            return

        dataset = point_cloud.data().copy()
        # if we have predicted something, we need to make the predicted labels the new actual labels
        if ('semclassidPredicted' in dataset.columns) or ('specificclassidPredicted' in dataset.columns):
            # remove the semclassid and specificclassid columns
            # ignore errors in order to not have to check whether the columns actually exist
            dataset.drop(columns=['semclassid', 'specificclassid'], inplace=True, errors='ignore')
            # rename the predicted columns to now be the actual labels
            dataset.rename(columns={'semclassidPredicted': 'semclassid', 'specificclassidPredicted': 'specificclassid'},
                           inplace=True)

        supported_columns_to_xpc_attribute_names = {v: k for k, v in self._xpc_to_supported_column_names.items()
                                                    if k in xpc_wrapper.supported_attributes}
        dataset.rename(supported_columns_to_xpc_attribute_names, axis='columns', inplace=True)
        attributes_map = {}
        python_wrapped_point_cloud = xpc_wrapper.PythonWrappedPointCloud()
        for column in dataset.columns:
            # the following first converts the float64 values (which is all of them) to float32
            # afterwards, those columns that can be expressed as integers will be converted to that
            # this is required für the xpc wrapper to properly save the data (without any precision errors or overflows)
            if column in xpc_wrapper.supported_attributes:
                attributes_map[column] = dataset[column].to_numpy().astype('float32')
                if np.array_equal(attributes_map[column], attributes_map[column].astype('int64')):
                    attributes_map[column] = attributes_map[column].astype('int64')
            else:
                logging.warning('Column "{}" is not supported by the xpc format and will not be'
                                ' written.'.format(str(column)))

        python_wrapped_point_cloud.attributes_map = attributes_map
        point_cloud_shift = (point_cloud.x_shift, point_cloud.y_shift, point_cloud.z_shift)
        python_wrapped_point_cloud.bounding_box_shift = point_cloud_shift
        xpc_wrapper.save(str(path), python_wrapped_point_cloud)


class LASFileHandler:
    def __init__(self):
        self.supported_las_formats = list(range(9))

        self.column_mapping = {'r': 'red', 'g': 'green', 'b': 'blue'}

        self.standard_field_defaults = {
            'return_number': 1,
            'number_of_returns': 1,
        }

        self.supported_extra_column_names = {
            'semclassid': 'int64',
            'specificclassid': 'int64',
            'semclassidpredicted': 'int64',
            'specificclassidpredicted': 'int64',
            'segmentid': 'int64',
            'segmentidpredicted': 'int64',
            'distancetodtm': 'float32'
        }

    @staticmethod
    def supported_types() -> List[str]:
        return ['.las', '.laz']


class LASReader(LASFileHandler, FileReader):
    """
    Reader for LAS/LAZ files.

    :param ignore_default_columns: Whether fields of an LAS/LAZ file that only contain default values should be ignored.
        Defaults to `True`.
    :type ignore_default_columns: bool, optional
    """

    def __init__(self, ignore_default_columns: bool = True):
        super().__init__()
        self.ignore_default_columns = ignore_default_columns

    def read_to_dataframe(self,
                          path: Union[str, Path],
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
        """Reads LAS/LAZ files with the columns ['x', 'y', 'z', 'intensity'] with no shift.

        :param path: path of the file to be read
        :type path: pathlib.Path or str
        :param columns: Point cloud columns / attributes to be read into memory. Defaults to `None`, which means that
            all columns are read.
        :type columns: List[str], optional
        :return: point cloud data as dataframe and shift
        :rtype: tuple(pandas.DataFrame, tuple(int)
        """
        las_data = laspy.read(path)
        data = np.array([las_data.x, las_data.y, las_data.z]).T
        point_cloud_df = pd.DataFrame(data, columns=['x', 'y', 'z'])

        for column_name in las_data.header.point_format.standard_dimension_names:
            if column_name in ['X', 'Y', 'Z']:
                continue
            values = np.array(las_data[column_name])
            default_value = self.standard_field_defaults.get(column_name, 0)
            if not self.ignore_default_columns or (values != default_value).any():
                point_cloud_df[column_name] = values

        for column_name in las_data.header.point_format.extra_dimension_names:
            point_cloud_df[column_name] = las_data[column_name]

        return point_cloud_df, (0, 0, 0)

    @staticmethod
    def read_max_resolutions(path: Union[str, Path]) -> Tuple[float, float, float]:
        """
        Reads the maximum resolution for each coordinate dimension from the header of the point cloud file.

        :param path: Path of the file to be read.
        :type path: pathlib.Path or str
        :return: Maximum resolution of the x-, y-, and z-coordinates of the point cloud.
        :rtype: Tuple[float, float, float]
        """

        with laspy.open(path, 'r') as f:
            scales = f.header.scales

        return scales


class LASWriter(LASFileHandler, FileWriter):
    """
    Writer for LAS/LAZ files.

    :param maximum_resolution: Maximum resolution of point coordinates in meter. Corresponds to the scale parameter
        used in the LAS/LAZ compression. Higher values result in a strong compression / lower point cloud resolution.
        Defaults to `1e-6`.
    :type maximum_resolution: float, optional
    """

    def __init__(self, maximum_resolution: float = 1e-6):
        super().__init__()
        self.maximum_resolution = maximum_resolution

    def select_point_format(self, point_cloud: pd.DataFrame) -> int:
        """
        Determines LAS file format that covers the most columns / attributes of a given point cloud.
        """
        columns = point_cloud.columns
        best_format = 0
        covered_columns = 0

        for f in self.supported_las_formats:
            current_format = laspy.point.format.PointFormat(f)
            columns_covered_by_current_format = len(set(current_format.standard_dimension_names).intersection(columns))
            if columns_covered_by_current_format > covered_columns:
                covered_columns = columns_covered_by_current_format
                best_format = current_format

        return best_format

    def write_to_file(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        """Writes point cloud to LAS/LAZ file.

        :param point_cloud: point cloud to be written
        :type point_cloud: PointCloud
        :param path: path of the file to be written
        :type path: pathlib.Path or str
        """
        point_cloud_df = point_cloud.data()
        point_cloud_df = point_cloud_df.rename(self.column_mapping, axis=1)

        las_data = laspy.create(point_format=self.select_point_format(point_cloud_df))
        point_coords = point_cloud_df[['x', 'y', 'z']].values
        offsets = point_coords.min(axis=0)
        scales = [self.maximum_resolution] * 3
        if point_cloud.x_max_resolution is not None:
            scales[0] = point_cloud.x_max_resolution
        if point_cloud.y_max_resolution is not None:
            scales[1] = point_cloud.y_max_resolution
        if point_cloud.z_max_resolution is not None:
            scales[2] = point_cloud.z_max_resolution

        las_data.change_scaling(scales=scales, offsets=offsets)
        las_data.xyz = point_coords

        for column_name in las_data.header.point_format.standard_dimension_names:
            if column_name in ['X', 'Y', 'Z']:
                continue

            if column_name.lower() in point_cloud_df.columns:
                las_data[column_name] = point_cloud_df[column_name.lower()]
            else:
                las_data[column_name] = np.full_like(las_data[column_name],
                                                     fill_value=self.standard_field_defaults.get(column_name, 0))

        for column_name, dtype in self.supported_extra_column_names.items():
            if column_name in point_cloud_df.columns:
                las_data.add_extra_dim(laspy.point.ExtraBytesParams(column_name, dtype))
                las_data.update_header()
                las_data[column_name] = point_cloud_df[column_name]

        las_data.write(path)


class NoFileHandler:
    @staticmethod
    def supported_types() -> List[str]:
        return ['.skip']


class NoReader(NoFileHandler, FileReader):
    def read(self,
             path: Union[str, Path],
             columns: Optional[List[str]] = None,
             use_smallest_types: bool = True) -> PointCloud:
        return PointCloud()


class NoWriter(NoFileHandler, FileWriter):
    def write(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        pass


class FileHandlerManager:
    """Superclass for managers. Classes that subclass this need to call this class' init function and pass a list of
    file readers/writers when initializing themselves.
    """

    def __init__(self, handlers: List[Any]) -> None:
        self._handlers = handlers
        self._supported_types: Dict[str, Type] = dict()
        self._init_supported_types()

    def _init_supported_types(self) -> None:
        for handler in self._handlers:
            for t in handler.supported_types():
                if t not in self._supported_types.keys():
                    self._supported_types[t] = handler

    def supported_types(self) -> Iterable[str]:
        return self._supported_types.keys()

    def add_handler(self, reader: Type) -> None:
        should_add = any(t not in self.supported_types() for t in reader.supported_types())
        if should_add:
            self._handlers.extend([reader])
            self._init_supported_types()
        else:
            logging.warning("{} was not added as all types can already be managed".format(reader.__name__))


class FileReaderManager(FileHandlerManager):
    """Manages all different :code:`FileReader` instances and is responsible for calling the right Reader when reading
    a file.
    """

    def __init__(self) -> None:
        handlers = [CSVReader, HDFReader, XPCReader, LASReader, NoReader]
        super(FileReaderManager, self).__init__(handlers)

    def read(self,
             path: Union[str, Path],
             columns: Optional[List[str]] = None, use_smallest_types: bool = True) -> PointCloud:
        """Delegates the read call to the correct reader and initializes the reader if necessary.

        :param path: path of the file to be read
        :type path: pathlib.Path or str
        :param columns: Point cloud columns / attributes to be read into memory. Defaults to `None`, which means that
            all columns are read.
        :type columns: List[str], optional
        :param use_smallest_datatypes: Whether for integer attributes the smallest possible data type should be used,
            with which all attribute values can be represented. Defaults to `True`.
        :type use_smallest_types: bool
        :return: read point cloud
        :rtype: PointCloud
        """
        p = Path(path)
        ending = p.suffix.lower()
        if ending not in self.supported_types():
            error_code = 'E705'
            logging.error("(Error {}) Tried to read file in an unsupported file format: {}.".format(error_code, ending))
            sys.exit(error_code)
        reader = self._supported_types[ending]
        if reader in self._handlers:
            reader = reader()
        point_cloud = reader.read(p, columns=columns, use_smallest_types=use_smallest_types)
        self._supported_types[ending] = reader
        return point_cloud


class FileWriterManager(FileHandlerManager):
    """Manages all different :code:`FileWriter` instances and is responsible for calling the correct Writer when writing
     a file.
    """

    def __init__(self):
        handlers = [CSVWriter, HDFWriter, XPCWriter, LASWriter, NoWriter]
        super(FileWriterManager, self).__init__(handlers)

    def write(self, point_cloud: PointCloud, path: Union[str, Path]) -> None:
        """Delegates the write call to the correct writer and initializes the writer if necessary.

        :param point_cloud: point cloud to be written
        :type point_cloud: PointCloud
        :param path: path of the file to be written
        :type path: pathlib.Path or str
        """
        p = Path(path)
        ending = p.suffix.lower()
        if ending not in self.supported_types():
            error_code = 'E706'
            logging.error("(Error {}) Tried to write file in an unsupported file format: {}.".format(error_code,
                                                                                                     ending))
            sys.exit(error_code)
        writer = self._supported_types[ending]
        if writer in self._handlers:
            writer = writer()
        writer.write(point_cloud, p)
        self._supported_types[ending] = writer
