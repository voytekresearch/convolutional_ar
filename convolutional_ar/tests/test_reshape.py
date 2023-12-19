"""Tests for reshaping."""

import numpy as np
import torch
from convolutional_ar.reshape import ar_reshape

def test_ar_reshape():
    # Test case 1: Check if the reshaped arrays have the correct shape
    x = np.random.rand(100, 100)
    order = 10
    reshaped_x, reshaped_y = ar_reshape(x, order)
    assert reshaped_x.shape == (100 * (100-order), order)
    assert reshaped_y.shape[0] == 100 * (100-order)

    # Test case 2: Check if the reshaped arrays have the correct values
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    order = 2
    reshaped_x, reshaped_y = ar_reshape(x, order)
    expected_x = np.array([[1, 2], [4, 5], [7, 8], [10, 11]])
    expected_y = np.array([3, 6, 9, 12])
    assert np.array_equal(reshaped_x, expected_x)
    assert np.array_equal(reshaped_y, expected_y)
