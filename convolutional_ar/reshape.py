"""Reshape an image or vector into an autoregressive problem."""

import numpy as np
import torch

def ar_reshape(x: np.ndarray, order: int) -> (np.ndarray, np.ndarray):
    """Lag a matrix of values to create an autoregressive problem.

    Parameters
    ----------
    x : 2d array
        Values (grayscale) of the image.
    order : int
        Number of points to lag.

    Returns
    -------
    x_ar : 2d array
        Past values, e.g (x_{i}, x_{i+1}... x_{i + order}).
    y_ar : 1d array
        Future values, e.g. (x_{i + order + 1}).
    """
    # AR problem
    x_ar = np.lib.stride_tricks.sliding_window_view(x, order, axis=1)[:, :-1]
    x_ar = x_ar.reshape(-1, order)
    y_ar = x[:, order:].reshape(-1)

    # To tensors
    x_ar = torch.from_numpy(x_ar.astype(np.float32))
    y_ar = torch.from_numpy(y_ar.astype(np.float32))

    return x_ar, y_ar

def ar_reshape_1d(x: np.ndarray, order: int) -> (np.ndarray, np.ndarray):
    """Lag a vector of values to create an autoregressive problem.

    Parameters
    ----------
    x : 1d array
        Vector values.
    order : int
        Number of points to lag.

    Returns
    -------
    x_ar : 2d array
        Past values, e.g (x_{i}, x_{i+1}... x_{i + order}).
    y_ar : 1d array
        Future values, e.g. (x_{i + order + 1}).
    """
    x_ar = np.lib.stride_tricks.sliding_window_view(x, order, axis=0)[:-1]
    y_ar = x[order:]
    return x_ar, y_ar