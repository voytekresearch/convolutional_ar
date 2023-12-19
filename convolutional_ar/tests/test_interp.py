"""Tests for interpolation."""
import pytest
import numpy as np
from convolutional_ar.interp import bilinear_interpolation


def test_bilinear_interpolation():
    img = np.array([[1, 2], [3, 4]])
    x_out = np.array([0, 1])
    y_out = np.array([0, 1])
    expected_output = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(bilinear_interpolation(img, x_out, y_out), expected_output)

def test_bilinear_interpolation_single_point():
    img = np.array([[1, 2], [3, 4]])
    x_out = np.array([0.5])
    y_out = np.array([0.5])
    expected_output = np.array([[2.5]])
    np.testing.assert_array_almost_equal(bilinear_interpolation(img, x_out, y_out), expected_output)

# def test_bilinear_interpolation_out_of_bounds():
#     # Todo: fix
#     img = np.array([[1, 2], [3, 4]])
#     x_out = np.array([-100, -100])
#     y_out = np.array([-100, -100])
#     with pytest.raises(IndexError):
#         bilinear_interpolation(img, x_out, y_out)