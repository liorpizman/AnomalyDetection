'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Tests for utils/helper methods
'''

import unittest

from utils.helper_methods import *


class TestGuiHelperMethods(unittest.TestCase):

    def test_valid_directory(self):
        """
        Test valid directory
        :return:
        """

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.assertEqual(is_valid_directory(dir_path), True)

    def test_invalid_directory(self):
        """
        Test invalid directory
        :return:
        """

        dir_path = 'invalid//directory'
        self.assertEqual(is_valid_directory(dir_path), False)

    def test_valid_cosine_similarity(self):
        """
        Test valid input
        :return:
        """

        x = np.array([0.1, 0.2])
        y = np.array([0.1, 0.4])

        self.assertEqual(cosine_similarity(x, y), 1 - (dot(x, y) / (norm(x) * norm(y))))

    def test_invalid_cosine_similarity(self):
        """
        Test invalid input
        :return:
        """

        x = np.array([0.1, 0.2])
        y = np.array([[0], [0]])

        x_transformed = x.reshape(-1, 1).transpose((1, 0))
        y_transformed = y.reshape(-1, 1).transpose((1, 0))

        self.assertEqual(cosine_similarity(x, y), 1 - pairwise.cosine_similarity(x_transformed, y_transformed)[0][0])

    def test_valid_euclidean_distance(self):
        """
        Test valid input
        :return:
        """

        x = np.array([2, 2])
        y = np.array([1, 1])

        self.assertEqual(euclidean_distance(x, y), np.linalg.norm(x - y))

    def test_valid_mahalanobis_distance(self):
        """
        Test valid input
        :return:
        """

        x = np.array([0.4, 0.8])
        y = np.array([0.4, 0.8])

        self.assertEqual(mahalanobis_distance(x, y), 0.0)

    def test_valid_mse(self):
        """
        Test valid input
        :return:
        """

        x = np.array([2.0, 3.0])
        y = np.array([1.0, 2.0])

        self.assertEqual(mse_distance(x, y), 1.0)
