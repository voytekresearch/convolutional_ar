"""Interpolation."""

import warnings
import numpy as np
from numba import jit, prange
from scipy import interpolate

def interpolate_circle(
    img: np.ndarray, n_points: int, radius: int, spacing: int,
    method: str='cubic'
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Interpolate a circular path around a center point

    Parameters
    ----------
    img : 2d array
        Values (grayscale) of the image.
    n_points : int
        Number of points to interpolate.
    radius : int
        Radius of the circle, in pixels.
    spacing : int
        Spacing between cicles, in pixels.

    Returns
    -------
    x2 : 1d array
        Interpolated x-coordinates.
    y2 : 1d array
        Interpolated y-coordinates.
    interp_vals : 1d array
        Interpolated values.
    """
    # Center points
    iy = np.arange(radius, len(img[0])-radius, spacing)
    ix = np.arange(radius, len(img)-radius, spacing)
    centers = np.array(np.meshgrid(iy, ix, copy=False)).T.reshape(-1, 2)

    # Circle
    theta_values = np.linspace(0, 2*np.pi, n_points)

    x_circle = np.cos(theta_values)
    y_circle = np.sin(theta_values)

    x_circle = x_circle - x_circle.min()
    x_circle = (x_circle / x_circle.max()) - 0.5

    y_circle = y_circle - y_circle.min()
    y_circle = (y_circle / y_circle.max()) - 0.5

    x_circle = x_circle * (radius * 2)
    y_circle = y_circle * (radius * 2)

    x2 = (x_circle + centers[:, 0][:, None])
    y2 = (y_circle + centers[:, 1][:, None])

    # Interpolate
    if method == 'bilinear':

        interp_vals = np.zeros((len(x2), n_points))
        for i, (x, y) in enumerate(zip(x2, y2)):
            interp_vals[i] = np.diag(bilinear_interpolation(img, x, y))

    elif method == 'cubic':
        x = np.arange(0, len(img[0]))
        y = np.arange(0, len(img))

        interp_vals = np.zeros((len(x2), n_points))
        interp = interpolate.RegularGridInterpolator((x, y), img, method='cubic')

        for i, (x_re, y_re) in enumerate(zip(x2, y2)):
            interp_vals[i] = interp((x_re, y_re))

    interp_vals = interp_vals.reshape(-1, n_points)
    x2 = x2.reshape(-1, n_points)
    y2 = y2.reshape(-1, n_points)

    return x2, y2, interp_vals


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bilinear_interpolation(
    img: np.ndarray, x_out: np.ndarray, y_out: np.ndarray
) -> np.ndarray:
    """Use for interpolation circles around center points.

    Parameters
    ----------
    img : 2d array
        Values (grayscale) of the image.
    x_out : 1d array
        Target x-coordinates.
    y_out : 1d array
        Target y-coordinates.

    Returns
    -------
    img_out : 2d array
        Interpoloated image.

    Notes
    -----

    - Taken from https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
    - Fix bug that returns nan somtimes

    """

    # Original coordinates
    h = len(img)
    w = len(img[0])
    x0 = np.arange(0, w).astype(np.float64)
    y0 = np.arange(0, h).astype(np.float64)

    # Interpolate
    img_out = np.zeros((y_out.size, x_out.size))

    for i in prange(img_out.shape[1]):
        idx = np.searchsorted(x0, x_out[i])

        x1 = x0[idx - 1]
        x2 = x0[idx]
        x = x_out[i]

        for j in prange(img_out.shape[0]):
            idy = np.searchsorted(y0, y_out[j])

            y1 = y0[idy - 1]
            y2 = y0[idy]
            y = y_out[j]

            f11 = img[idy - 1, idx - 1]
            f21 = img[idy - 1, idx]
            f12 = img[idy, idx - 1]
            f22 = img[idy, idx]

            img_out[j, i] = (
                f11 * (x2 - x) * (y2 - y) +
                f21 * (x - x1) * (y2 - y) +
                f12 * (x2 - x) * (y - y1) +
                f22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1))

    return img_out
