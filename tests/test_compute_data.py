import math
import os
from unittest.mock import Mock
import pytest

import numpy as np
from inflammation import models
from inflammation.compute_data import CSVDataSource, analyse_data, compute_standard_deviation_by_day

def test_analyse_data_mock_source():
  from inflammation.compute_data import analyse_data
  data_source = Mock()

  data_source.load_inflammation_data.return_value = [
    np.array([[1, 2], [3, 4], [5, 6]]),
    np.array([[2, 3], [4, 5], [6, 7]]),
  ]

  analyse_data(data_source)


def test_analyse_data():
    path = os.path.join(os.path.dirname(__file__), "../data")
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64

@pytest.mark.parametrize(
  'data, expected_output', [
    (
       [[[0, 1, 0], [0, 2, 0]]],
       [0, 0, 0]
    ),
    (
       [[[0, 2, 0]], [[0, 1, 0]]],
       [0, math.sqrt(0.25), 0]
    ),
    (
       [[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]],
       [0, 0, 0]
    )
])
def test_compute_standard_deviation_by_day(data, expected_output):
    result = compute_standard_deviation_by_day(data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected_output)

def test_compute_standard_deviation_by_day_take_CSVDataSource():
    path = os.path.join(os.path.dirname(__file__), "../data")
    data_source = CSVDataSource(path)
    result = compute_standard_deviation_by_day(data_source.load_inflammation_data())
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
