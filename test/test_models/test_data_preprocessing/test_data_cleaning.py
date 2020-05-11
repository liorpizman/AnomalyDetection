'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Tests for models/data_preprocessing/data_cleaning
'''

import unittest

from models.data_preprocessing.data_cleaning import *


class TestModelsDataPreprocessingDataCleaning(unittest.TestCase):

    def test_valid_clean_data(self):
        """
        Test clean data for valid data frame
        :return:
        """

        data = [[13.0, float("nan")], [float("nan"), 10.0], [14.0, float("nan")]]

        df = pd.DataFrame(data, columns=['GPS DISTNACE', 'label'])

        desired_data = [[13.0], [13.0], [14.0]]

        desired_df = pd.DataFrame(desired_data, columns=['GPS DISTNACE'])

        self.assertEqual(desired_df.equals(clean_data(df)), True)

    def test_empty_clean_data(self):
        """
        Test clean data for empty data frame
        :return:
        """

        df = pd.DataFrame()

        desired_df = pd.DataFrame()

        self.assertEqual(desired_df.equals(clean_data(df)), True)

    def test_drop_columns(self):
        """
        Test drop columns for valid data frame
        :return:
        """

        df = pd.DataFrame(columns=['LATITUDE', 'Drone IP'])

        desired_df = pd.DataFrame(columns=['LATITUDE'])

        self.assertEqual(desired_df.equals(drop_columns(df, 'Drone IP')), True)

    def test_empty_drop_columns(self):
        """
        Test drop columns for empty data frame
        :return:
        """

        df = pd.DataFrame()

        desired_df = pd.DataFrame()

        self.assertEqual(desired_df.equals(drop_columns(df, 'Drone IP')), True)

    def test_valid_remove_na(self):
        """
        Test remove na for valid data frame
        :return:
        """

        data = [[1.0], [float("nan")], [3.0]]

        df = pd.DataFrame(data)

        desired_data = [[1.0], [1.0], [3.0]]

        desired_df = pd.DataFrame(desired_data)

        self.assertEqual(desired_df.equals(remove_na(df)), True)

    def test_empty_remove_na(self):
        """
        Test remove na for empty data frame
        :return:
        """

        df = pd.DataFrame()

        desired_df = pd.DataFrame()

        self.assertEqual(desired_df.equals(remove_na(df)), True)
