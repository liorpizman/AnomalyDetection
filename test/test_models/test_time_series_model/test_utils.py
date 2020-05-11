'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Tests for models/time_series_model/utils
'''

import unittest

from models.time_series_model.utils import *


class TestModelsTimeSeriesModelUtils(unittest.TestCase):

    def test_valid_safe_shape(self):
        """
        Test valid input
        :return:
        """

        input_array = np.array([[2], [3], [4], [1], [2], [3]])
        input_index = 1

        expected_output = 1
        actual_output = safe_shape(input_array, input_index)

        self.assertEqual(expected_output, actual_output)

    def test_positive_invalid_safe_shape(self):
        """
        Test positive invalid input
        :return:
        """

        input_array = np.array([[2], [3], [4]])
        input_index = 7

        expected_output = 1
        actual_output = safe_shape(input_array, input_index)

        self.assertEqual(expected_output, actual_output)

    def test_negative_invalid_safe_shape(self):
        """
        Test negative invalid input
        :return:
        """

        input_array = np.array([[2], [3], [4]])
        input_index = -4

        try:
            safe_shape(input_array, input_index)
            assert False

        except IndexError:
            assert True

    def test_valid_raw_values_mse(self):
        """
        Test valid input using raw values
        :return:
        """

        X1 = np.array([[1, 2], [1, 3]])
        X2 = np.array([[1, 2], [3, 1]])

        expected_output = np.array([1.41421356, 1.41421356])
        actual_output = mse(X1, X2, multi_output='raw_values')

        self.assertEqual(expected_output.all(), actual_output.all())

    def test_valid_uniform_average_mse(self):
        """
        Test valid input using uniform average
        :return:
        """

        X1 = np.array([[1, 2], [1, 3]])
        X2 = np.array([[1, 2], [3, 1]])

        expected_output = np.array([1.41421356, 1.41421356])
        actual_output = mse(X1, X2, multi_output='uniform_average')

        self.assertEqual(expected_output.all(), actual_output.all())

    def test_empty_raw_values_mse(self):
        """
        Test empty input using raw values
        :return:
        """

        X1 = np.array([[], []])
        X2 = np.array([[], []])

        expected_output = np.array([[]])
        actual_output = mse(X1, X2, multi_output='raw_values')

        self.assertEqual(expected_output.all(), actual_output.all())

    def test_empty_uniform_average_mse(self):
        """
        Test empty input using uniform average
        :return:
        """

        X1 = np.array([[], []])
        X2 = np.array([[], []])

        expected_output = np.array([[]])
        actual_output = mse(X1, X2, multi_output='uniform_average')

        self.assertEqual(expected_output.all(), actual_output.all())

    def test_valid_raw_values_multi_mse(self):
        """
        Test valid input using uniform average
        :return:
        """
        """
        Test valid input using raw values
        :return:
        """

        X1 = np.array([[[1, 2], [1, 3]]])
        X2 = np.array([[[1, 2], [3, 1]]])

        expected_output = np.array([1.41421356, 1.41421356])
        actual_output = multi_mse(X1, X2, multi_output='raw_values')

        self.assertEqual(expected_output.all(), actual_output.all())

    def test_valid_uniform_average_multi_mse(self):
        """
        Test valid input using uniform average
        :return:
        """

        X1 = np.array([[[1, 2], [1, 3]]])
        X2 = np.array([[[1, 2], [3, 1]]])

        expected_output = np.array([1.41421356, 1.41421356])
        actual_output = multi_mse(X1, X2, multi_output='uniform_average')

        self.assertEqual(expected_output.all(), actual_output.all())

    def test_empty_raw_values_multi_mse(self):
        """
        Test empty input using raw values
        :return:
        """

        X1 = np.array([[[], []]])
        X2 = np.array([[[], []]])

        expected_output = np.array([[]])
        actual_output = mse(X1, X2, multi_output='raw_values')

        self.assertEqual(expected_output.all(), actual_output.all())

    def test_empty_uniform_average_multi_mse(self):
        """
        Test empty input using uniform average
        :return:
        """

        X1 = np.array([[[], []]])
        X2 = np.array([[[], []]])

        expected_output = np.array([[]])
        actual_output = mse(X1, X2, multi_output='uniform_average')

        self.assertEqual(expected_output.all(), actual_output.all())
