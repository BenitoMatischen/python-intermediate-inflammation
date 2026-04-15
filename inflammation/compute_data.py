"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views


class CSVDataSource:
    """Class for loading inflammation data from CSV files within a directory."""
    def __init__(self, data_dir : str) -> None:
        self.__data_dir = data_dir

    def load_inflammation_data(self) -> list[np.ndarray]:
        """Load all inflammation data from CSV files within the directory.

        :returns: List of 2D Numpy arrays containing inflammation data
        """
        data_file_paths = glob.glob(os.path.join(self.__data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.__data_dir}")
        return [np.array(models.load_csv(file_path)) for file_path in data_file_paths]


def analyse_data(data_source : CSVDataSource) -> None:
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    
    data = data_source.load_inflammation_data()

    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    views.visualize(graph_data)