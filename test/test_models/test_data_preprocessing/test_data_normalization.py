'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Tests for models/data_preprocessing/data_normalization
'''

import unittest

import pandas as pd
from numpy import array_equal

from models.data_preprocessing.data_normalization import *


class TestModelsDataPreprocessingDataNormalization(unittest.TestCase):

    def test_valid_normalize_data(self):
        """
        Test normalize data for valid data frame
        :return:
        """

        data = [[1.0], [2.0], [3.0]]

        df = pd.DataFrame(data)

        desired_data = pd.np.array([[0.0], [0.5], [1.0]])

        self.assertEqual(desired_data.all(), normalize_data(df, "min_max")[0].all())

    def test_invalid_normalize_data(self):
        """
        Test normalize data for invalid scaler name
        :return:
        """

        data = [[1.0], [2.0], [3.0]]

        df = pd.DataFrame(data)

        desired_data = pd.np.array([[-1.22474487], [0.0], [-1.22474487]])

        self.assertEqual(desired_data.all(), normalize_data(df, "invalid_scaler")[0].all())

    def test_empty_normalize_data(self):
        """
        Test normalize data for empty data frame
        :return:
        """

        data = [[]]

        df = pd.DataFrame(data)

        try:
            normalize_data(df, "min_max")
            assert False

        except ValueError:
            assert True
