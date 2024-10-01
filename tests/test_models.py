"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
        "test, expected",
        [
            ([ [0, 1], [10, -2], [1, 4], [6, 7], [10, 7]], [10, 7]),
            ([ [0, 0], [1, 1], [2, 2]], [2, 2])
        ]
)
def test_daily_max(test, expected):
    """Test that max function works for an array of integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
        "test, expected",
        [
            ([[0, 1], [10, -2], [1, 4], [6, 7]], [0, -2]),
            ([[5, 4], [3, 2], [1, 0]], [1, 0])
        ]
)
def test_daily_min(test, expected):
    """Test that min function works for an array of integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))