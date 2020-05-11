'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Tests for gui/shared/helper methods
'''

import unittest

from gui.shared.helper_methods import *
from pathlib import Path

class TestGuiSharedHelperMethods(unittest.TestCase):

    def test_valid_load_classification_methods(self):
        """
        Test valid load classification methods
        :return:
        """

        anomaly_detection_methods = load_classification_methods('anomaly_detection_methods')
        anomaly_detection_list = ['LSTM', 'SVR', 'MLP', 'Random Forest']

        self.assertEqual(anomaly_detection_methods, anomaly_detection_list)

    def test_empty_load_classification_methods(self):
        """
        Test empty list
        :return:
        """

        anomaly_detection_methods = load_classification_methods('')

        self.assertEqual(anomaly_detection_methods, None)

    def test_invalid_load_classification_methods(self):
        """
        Test invalid list
        :return:
        """

        anomaly_detection_methods = load_classification_methods('anomaly_detection_pre_processing_methods')

        self.assertEqual(anomaly_detection_methods, None)

    def test_load_anomaly_detection_list(self):
        """
        Test valid list
        :return:
        """

        anomaly_detection_list = ['LSTM', 'SVR', 'MLP', 'Random Forest']
        anomaly_detection_methods = load_anomaly_detection_list()

        self.assertEqual(anomaly_detection_methods, anomaly_detection_list)

    def test_load_similarity_list(self):
        """
        Test valid list
        :return:
        """

        similarity_functions_list = ['Cosine similarity', 'Mahalanobis distance', 'MSE', 'Euclidean distance']
        similarity_functions = load_similarity_list()

        self.assertEqual(similarity_functions, similarity_functions_list)

    def test_valid_read_json_file(self):
        """
        Test valid JSON file path
        :return:
        """

        json_data = {'features': 'GPS Distance'}
        json_file_path = 'model_data.json'

        ROOT_DIR = Path(__file__).parent.parent.parent
        full_path = os.path.join(*[str(ROOT_DIR), 'test_gui', 'test_shared', json_file_path])

        self.assertEqual(read_json_file(full_path), json_data)

    def test_invalid_read_json_file(self):
        """
        Test invalid JSON file path
        :return:
        """

        invalid_json_file_path = './invalid.json'

        try:
            read_json_file(invalid_json_file_path)
            assert False

        except FileNotFoundError:
            assert True

    def test_valid_trim_unnecessary_chars(self):
        """
        Test valid input
        :return:
        """

        text_input = 'random_forest'
        expected_output = 'Random Forest'
        actual_output = trim_unnecessary_chars(text_input)

        self.assertEqual(expected_output, actual_output)

    def test_invalid_trim_unnecessary_chars(self):
        """
        Test invalid input
        :return:
        """

        text_input = 'my_purpose_is_to_be_random'
        expected_output = 'My purpose is to be random'
        actual_output = trim_unnecessary_chars(text_input)

        self.assertEqual(expected_output, actual_output)

    def test_valid_transform_list(self):
        """
        Test valid input
        :return:
        """

        text_input = ['random_forest', 'cosine_similarity']
        expected_output = ['Random Forest', 'Cosine similarity']
        actual_output = transform_list(text_input)

        self.assertEqual(expected_output, actual_output)

    def test_empty_transform_list(self):
        """
        Test empty input
        :return:
        """

        text_input = []
        expected_output = []
        actual_output = transform_list(text_input)

        self.assertEqual(expected_output, actual_output)

    def test_valid_clear_text(self):
        """
        Test valid input
        :return:
        """

        expected_output = ''
        try:
            clear_text(expected_output)
            actual_output = ''
        except:
            actual_output = ''

        self.assertEqual(expected_output, actual_output)

    def test_invalid_clear_text(self):
        """
        Test invalid input
        :return:
        """

        try:
            clear_text(None)
            assert False

        except AttributeError:
            assert True
