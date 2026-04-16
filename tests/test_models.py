"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean
from inflammation.compute_data import CSVDataSource

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

@pytest.mark.parametrize(
    "test_input, expected",
    [
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4])),
    ]
)
def test_load_inflammation_data(test_input, expected):
    data_source = CSVDataSource('data/')
    data = data_source.load_inflammation_data()
    assert len(data) == 12
    "test, expected, expect_raises",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None
        ),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            None
        ),
        (
            [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]],
            [[-0.33, -0.67, -1], [-0.67, -0.83, -1], [-0.78, -0.89, -1]],
            None
        ),
        (
            [[1, 2], [3, 4], [5, 6]],
            None,
            ValueError
        )
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(np.array(test))
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
