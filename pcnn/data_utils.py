"""Module providing utility functions transforming data. All functions are only used in the PointCloudDataset."""

__all__ = ['jitter', 'random_rotate', 'random_scale', 'shift', 'translate_scale_xy', 'translate_scale_xyz']

from typing import Optional

import numpy as np


def random_rotate(points: np.ndarray, normal_index: int, generator: Optional[np.random.Generator] = None) -> np.ndarray:
    """Randomly rotates an array of points and its corresponding normals

    :param points: array of points
    :type points: np.ndarray
    :param normal_index: start index for the three per point normals if available
    :type normal_index: integer
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    :return: array of randomly rotated points
    :rtype: np.ndarray
    """
    if generator is None:
        generator = np.random.default_rng()

    rotation_angle = generator.uniform(0, 1) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    if isinstance(points, list):
        points = [np.copy(point) for point in points]
        for point in points:
            point[:, :3] = np.dot(point[:, :3].reshape((-1, 3)), rotation_matrix)
        return points
    points = np.copy(points)
    points[:, :3] = np.dot(points[:, :3].reshape((-1, 3)), rotation_matrix)

    # normals
    if normal_index != -1:
        points[:, normal_index:normal_index + 3] = np.dot(points[:, normal_index:normal_index + 3].reshape((-1, 3)),
                                                          rotation_matrix)
    return points


def translate_scale_xyz(points: np.ndarray, center_point: np.ndarray, scale_factor: float) -> np.ndarray:
    """ Translates the xyz-coordinates of the points so that the specified center point becomes the coordinate origin.
    Then scales the coordinates by the specified scale factor.

    :param points: array of points
    :type points: np.ndarray
    :param center_point: Center point to be translated to the coordinate origin.
    :type center_point: np.ndarray
    :param scale_factor: Scale factor
    :type scale_factor: float
    :return: array of scaled points
    :rtype: np.ndarray
    """
    points = np.copy(points)
    points[:, :3] -= center_point[:3]
    points[:, :3] *= scale_factor
    return points


def translate_scale_xy(points: np.ndarray, center_point: np.ndarray, scale_factor: float) -> np.ndarray:
    """ Translates the xy-coordinates of the points so that the specified center point becomes the coordinate origin.
    Then scales the coordinates by the specified scale factor.

    :param points: array of points
    :type points: numpy.ndarray
    :param center_point: Center point to be translated to the coordinate origin.
    :type center_point: np.ndarray
    :param scale_factor: Scale factor
    :type scale_factor: float
    :return: array of scaled points
    :rtype: numpy.ndarray
    """
    points = np.copy(points)
    points[:, :2] -= center_point[:2]
    points[:, 2] -= np.quantile(points[:, 2], q=0.05)
    points[:, :3] *= scale_factor
    return points


def jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05, radius: float = 3,
           generator: Optional[np.random.Generator] = None) -> np.ndarray:
    """Shifts every point individually slightly, in a random direction by random amount.

    :param points: array of points
    :type points: np.ndarray
    :param sigma: strength of the shifts
    :type sigma: float
    :param clip: maximum shift per dimension
    :type clip: float
    :param radius: original radius of the point, adjusts how far the points are moved.
    :type radius: float
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    :return: array of shifted points
    :rtype: np.ndarray
    """
    if generator is None:
        generator = np.random.default_rng()

    points = np.copy(points)
    N, _ = points.shape
    assert radius > 0
    assert clip > 0

    clip *= 3 / radius
    sigma *= 3 / radius
    jitter_value = np.clip(sigma * generator.standard_normal(size=(N, 3)), -1 * clip, clip)
    points[:, :3] += jitter_value
    return points


def shift(points: np.ndarray, sigma: float = 0.05, generator: Optional[np.random.Generator] = None) -> np.ndarray:
    """Shifts all the points together randomly

    :param points: array of points
    :type points: np.ndarray
    :param sigma: strength of the shift
    :type sigma: float
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    :return: array of shifted points
    :rtype: np.ndarray
    """
    if generator is None:
        generator = np.random.default_rng()

    arr = np.zeros(points.shape[1])
    shifts = generator.normal(scale=sigma, size=3)
    arr[:3] += shifts
    return points + arr


def random_scale(points: np.ndarray, sigma: float = 0.1, generator: Optional[np.random.Generator] = None) -> np.ndarray:
    """Scales all the points coordinates randomly

    :param points: array of points
    :type points: np.ndarray
    :param sigma: strength of the scaling
    :type sigma: float
    :param generator: If not None, this random number generator will be used. Defaults to `None`.
    :type generator: numpy.random.Generator, optional
    :return: array of scaled points
    :rtype: np.ndarray
    """
    if generator is None:
        generator = np.random.default_rng()

    points = np.copy(points)
    rand = generator.normal(scale=sigma, size=3)
    points[:, :3] *= (1 + rand)
    return points
