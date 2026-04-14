"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest

from inflammation.models import daily_mean, daily_max, daily_min


@pytest.mark.parametrize("test_input, expected", [
    (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
    (np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4]))
    ]
)
def test_daily_mean_integers(test_input, expected):
    """Test that mean function works for an array of positive integers."""
    
    npt.assert_array_equal(daily_mean(test_input), expected)

def test_daily_max():
    """Test that max function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([5, 6])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_min():
    """Test that min function works for an array of positive integers."""

    test_input = np.array([[1, 2, -4],
                           [3, 4, -2],
                           [5, -22, -1]])
    test_result = np.array([1, -22, -4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)