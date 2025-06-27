"""Module providing general utility functions meant to be used throughout the project as needed.
"""
import argparse
import hashlib
import logging
from logging.handlers import QueueHandler
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Queue
import random
import sys
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from scipy.spatial.kdtree import KDTree
import torch
from tqdm import tqdm

from pcnn import globals

def _triangular_number(x: int) -> int:
    """Calculates the triangular number (https://en.wikipedia.org/wiki/Triangular_number).

    :param x: number to calculate triangular number for
    :type x: number
    :return: the calculated triangular number
    :rtype: number
    """
    return int((x * (x + 1)) / 2)


def _triangular_root(x: int) -> int:
    """Calculates the triangular root
    (https://en.wikipedia.org/wiki/Triangular_number#Triangular_roots_and_tests_for_triangular_numbers).

    :param x: number to calculate triangular number for
    :type x: number
    :return: the calculated triangular number
    :rtype: number
    """
    return int(math.trunc((math.sqrt(8 * x + 1) - 1) / 2))


def pair_function(x: int, y: int) -> int:
    """Calculates the Cantor pairing function (https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function)
    to compute the unique_id from the baseclassid and specificclassid.

    :param x: baseclassid
    :type x: unsigned integer
    :param y: specificclassid
    :type y: unsigned integer
    :return: the unique_id
    :rtype: unsigned integer
    """
    return int(x + 0.5 * (x + y) * (x + y + 1))


def reverse_unique_id(unique_id: int) -> Tuple[int, int]:
    """Reverses the Cantor pairing function (https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function)
    to compute the semclassid and specificclassid from an unique_id.

    :param unique_id: unique_id to calculate semclassid and specificclassid from
    :type unique_id: number
    :return: a tuple consisting of semclassid and specificclassid
    :rtype: tuple of two numbers
    """
    if unique_id == -1:
        return -1, 0
    else:
        base_sem_class = int(unique_id) - _triangular_number(_triangular_root(unique_id))
        specific_sem_class = _triangular_root(unique_id) - base_sem_class
        return int(base_sem_class), int(specific_sem_class)


def string_to_class(current_class_name: str, classname: str) -> Type:
    """Gets and returns a class based on the module it is included into and its name.

    :param current_class_name: name of a module including the class to be returned
    :type current_class_name: string
    :param classname: name of the class to be returned
    :type classname: string
    :return: class corresponding to the classname parameter if included in the current_class_name module
    :rtype: class
    """
    return getattr(sys.modules[current_class_name], classname)


def overwrite_settings_if_present(settings: Dict[str, Any], key: str, overwrite_argument: Any) -> None:
    """Sets a value in the given dictionary if it's key is already existing.

    :param settings: settings dictionary to be modified
    :type settings: dict
    :param key: key in the dictionary
    :type key: string
    :param overwrite_argument: new value to be written
    :type overwrite_argument: arbitrary
    :return: nothing
    """
    # typesafe nan check
    # will only result in True if overwrite_argument is nan
    if isinstance(overwrite_argument, list):
        for argument in overwrite_argument:
            if argument != argument:
                overwrite_argument = None
                break
    else:
        if overwrite_argument != overwrite_argument:
            overwrite_argument = None
    if overwrite_argument is not None:
        if isinstance(overwrite_argument, argparse.Namespace):
            overwrite_argument = vars(overwrite_argument)
        if key in settings.keys():
            settings[key] = overwrite_argument
        else:
            logging.warning("Key \"{}\" was not present in the given settings dictionary. Skipping value insertion."
                            .format(key))


def start_logging_in_worker_process(logging_queue: Queue, log_level: str) -> None:
    """
    Configures logging inside a worker process.

    :param logging_queue: Multiprocessing queue to which the worker process should pass its log messages.
    :type logging_queue: multiprocessing.Queue
    :param log_level: logging library's level of detail for log prints
    :type log_level: string
    """

    queue_handler = QueueHandler(logging_queue)
    queue_handler.setLevel(log_level)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(queue_handler)


def start_logging(log_file_path: Path, log_level: str, append: bool = True) -> Union[int, str]:
    """Configures and starts logging for the project using the logging library.

    :param log_file_path: path to log file
    :type log_file_path: pathlib.Path
    :param log_level: logging library's level of detail for log prints
    :type log_level: string
    :param append: determines if log file should be opened in append mode, defaults to True
    :type append: bool
    :return: 0 on success, error code if exception occurs
    :rtype: Union[int, str]
    """
    try:
        # on Linux and MacOS only the 'fork' method works, while on Windows only 'spawn' works
        mp_context: Union[mp.context.SpawnContext, mp.context.ForkContext]
        mp_context = mp.get_context('spawn') if 'win' in sys.platform else mp.get_context('fork')
        logging_manager = mp_context.Manager()
        logging_queue: Queue = logging_manager.Queue()

        # ensure that the log file can be created before starting the logger process
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        f = open(log_file_path, 'a')
        f.close()
        logging_process: mp.process.BaseProcess = mp_context.Process(target=logger_process,
                                                                     args=(logging_queue, log_file_path, log_level,
                                                                           append))
        logging_process.start()

        logger = logging.getLogger()
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(log_level)

        globals.logging_manager = logging_manager
        globals.logging_queue = logging_queue
        globals.logging_process = logging_process
    except Exception:
        reset_logging()

        return "E701"
    return 0


def reset_logging() -> None:
    """ Closes all open logging files and resets the handlers to allow for logging to be re-initialized with new
    targets.
    """
    if globals.logging_queue is not None:
        globals.logging_queue.put_nowait(None)
    if globals.logging_process is not None:
        globals.logging_process.join()

    for handler in logging.getLogger().handlers:
        handler.close()
    logging.getLogger().handlers = []

    if globals.logging_manager is not None:
        globals.logging_manager.shutdown()

    globals.logging_queue = None
    globals.logging_process = None
    globals.logging_manager = None


def logger_process(queue: Queue, log_file_path: Path, log_level: str, append: bool = False):
    """
    Logger process that coordinates / synchronizes logging between multiple worker processes.

    For more details, see https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-
    processes

    :param queue: Queue of logging messages from multiple worker processes.
    :type queue: multiprocessing.Queue
    :param log_file_path: path to log file
    :type log_file_path: pathlib.Path
    :param log_level: logging library's level of detail for log prints
    :type log_level: string
    :param append: determines if log file should be opened in append mode, defaults to True
    :type append: bool
    """

    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_mode = 'a' if append else 'w'
    file_handler = logging.FileHandler(log_file_path, mode=file_mode)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(log_level)

    while True:
        message = queue.get()
        if message is None:  # check for shutdown
            break
        logger.handle(message)


def get_length(length: int, k: int, length_divider: int) -> int:
    """Function to calculate the number of samples that a :code:`DataLoader` should have. This is a heuristic
    that was tuned experimentally.

    :param length: length of the dataset (number of points)
    :type length: int
    :param k: number of points sampled in a radius query
    :type k: int
    :param length_divider: parameter for tuning the length of the dataset
    :type length_divider: int
    :return: length of loader
    :rtype: int
    """
    return int(np.ceil(length / (length_divider * (k / 2048))))


def tqdm_(loader, gui: bool = False) -> tqdm:
    """Wrapper for tqdm (progress bar) to make the main code less cluttered.

    :param loader: point cloud loader object
    :type loader: MultiPointCloudLoader
    :param gui: whether the GUI is used or not, if True printing of progress is suppressed
    :type gui: bool
    :return: progress bar object
    :rtype: tqdm
    """
    # while disable = true will disable all of the progress reporting, we know the total amount of iterations and
    # thus can calculate the progress ourselves
    return tqdm(loader, ascii=True, leave=False, file=sys.stdout, disable=gui)


def predict_tqdm(gui: bool = False) -> tqdm:
    """Wrapper for tqdm (progress bar) for the prediction phase.

    :param gui: whether the GUI is used or not, if True printing of progress is suppressed
    :type gui: bool
    :return: progress bar object
    :rtype: tqdm
    """
    return tqdm(total=100, ascii=True, leave=False, unit_scale=True, file=sys.stdout, disable=gui)


def step_tqdm(steps: int, gui: bool = False) -> tqdm:
    """Wrapper for tqdm (progress bar) for that allows to specify a total number of steps.

    :param num_samples: Total number of steps.
    :type num_samples: integer
    :param gui: whether the GUI is used or not. If `True`, printing of progress is suppressed.
    :type gui: bool
    :return: progress bar object
    :rtype: tqdm
    """
    return tqdm(total=steps, ascii=True, leave=False, unit_scale=True, file=sys.stdout, disable=gui)


def worker_init_fn(worker_id) -> None:
    """
    Needed to make sure that Torch Dataloaders produce different random numbers when using multiple threads:
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    https://github.com/pytorch/pytorch/issues/5059
    """

    worker_info = torch.utils.data.get_worker_info()
    worker_seed = worker_info.seed % 2**32
    worker_dataset = worker_info.dataset

    # reseed random generator with the worker seed so that each copy of the dataset returns different samples
    if type(worker_dataset).__name__ == 'PointCloudDataset':
        worker_dataset.random_generator = np.random.default_rng(worker_seed)
        # TODO: don't use `hasattr` as all datasets should have some sort of point_sampler
        if hasattr(worker_dataset, 'point_sampler'):
            worker_dataset.point_sampler.generator = worker_dataset.random_generator

    # replace seed by the worker seed so that each copy of the dataset returns different samples
    if type(worker_dataset).__name__ == 'PredictionPointCloudDataset':
        worker_dataset.seed = worker_seed

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_gui_active() -> bool:
    """Checks for existence of the :code:`PCNN_USES_GUI` environment variable.

    :return: True if PCNN_USES_GUI env var exists
    :rtype: bool
    """
    return 'PCNN_USES_GUI' in os.environ.keys()


def report_progress(progress: float) -> None:
    """Passes a percentage progress value (float) to the GUI progress bar and updates the UI.

    :param progress: progress as float (0-1)
    :type progress: float
    :return: nothing
    """
    from PySide2.QtCore import QCoreApplication
    globals.progress_bridge.set_progress(progress * 100)
    QCoreApplication.processEvents()


def set_progress_phase(phase) -> None:
    """Accesses the global :code:`progress_bridge` to tell the GUI which phase (preprocess, train, test, predict)
    the execution is currently in.

    :param phase: phase identifier
    :type phase: ProgressBridge.Phases
    :return: nothing
    """
    from PySide2.QtCore import QCoreApplication
    globals.progress_bridge.set_phase(phase)
    QCoreApplication.processEvents()


def get_pcnn_root_dir() -> Optional[Path]:
    cwd = Path.cwd()
    pcnn_dir = None
    scripts_in_pcnn = {'benchmark.py', 'preprocess.py', 'train.py', 'predict.py'}

    while cwd.parent != cwd:
        script_dir = cwd / 'scripts'
        if len(scripts_in_pcnn.difference(set(file.name for file in script_dir.glob('*.py')))) == 0:
            pcnn_dir = cwd
            break
        cwd = cwd.parent

    return pcnn_dir


def knn(source: np.ndarray, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    tree = KDTree(source, compact_nodes=False, balanced_tree=False)
    if len(source) < k:
        logging.warning(f"Tried to find {k} neighbors in {len(source)} points. Returning only {len(source)}")
        k = len(source)
    dists, indices = tree.query(query, k)
    return dists, indices


def set_deterministic_mode(level: Literal['none', 'basic', 'full'], seed: int = 0) -> None:
    """
    This method allows to choose between different levels of determinism and reproducibility. For more details, see
    https://pytorch.org/docs/stable/notes/randomness.html

    :param level: Level of determinism to be used: `'none'` | `'basic'` | `'full'`.

        - `'none'`: Data sampling, model initialization, the selection of CUDA algorithms and the selected CUDA
          algorithms themselves can be non-deterministic.
        - `'basic'`: Data sampling, model initialization, and the selection of CUDA algorithms are deterministic, while
          the selected CUDA algorithms themselves can be non-deterministic.
        - `'full'`: Data sampling, model initialization, the selection of CUDA algorithms and the selected CUDA
          algorithms themselves are deterministic. In this mode, training and prediction results are fully reproducible,
          but performance can be degraded significantly.
    :type level: string
    :param seed: Seed to be used in random processes. Defaults to 0.
    :type seed: int, optional.
    """

    if level not in ['none', 'basic', 'full']:
        raise ValueError(f"Invalid level of determinism: {level}.")

    globals.deterministic_mode = level

    if level == 'none':
        torch.seed()
        random.seed(None)
        np.random.seed(None)

        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
    else:
        # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # seed random processes
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # see https://pytorch.org/docs/stable/notes/randomness.html
        # this ensures that cuDNN deterministically selects the algorithms
        torch.backends.cudnn.benchmark = False
        # this ensures that only deterministic algorithms are selected when level is set to 'full'
        torch.use_deterministic_algorithms(level == 'full')


def hash_dataset(dataset: Union[pd.DataFrame, np.ndarray]) -> str:
    """
    Hashes parts of a dataset.

    :param dataset: Dataset for which a hash is to be computed.
    :type dataset: Union[pandas.DataFrame, numpy.ndarray].
    :return: Hash of the dataset.
    :rtype: string.
    """

    if isinstance(dataset, pd.DataFrame):
        hash_string = ''.join(str(x) for x in [dataset.shape, dataset.head(2), dataset.tail(2)])
    else:
        hash_string = ''.join(str(x) for x in [dataset.shape, dataset[:5], dataset[-5:]])
    return hashlib.md5(hash_string.encode('utf-8')).hexdigest()[:10]
