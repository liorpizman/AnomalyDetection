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

    def setUp(self):
        """
        Set up values before test
        :return:
        """

        self.prediction = [0, 1, 1]
        self.attack_start = 1
        self.attack_end = 2
        self.add_window_size = True
        self.window_size = 1

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

    def test_invalid_shape_cosine_similarity(self):
        """
        Test invalid shape for cosine similarity method
        :return:
        """

        x = np.array([0.1, 0.2])
        y = np.array([5])

        try:
            cosine_similarity(x, y)
            assert False

        except ValueError:
            assert True

    def test_invalid_shape_euclidean_distance(self):
        """
        Test invalid shape for euclidean distance method
        :return:
        """

        x = np.array([0.1, 0.2])
        y = np.array(["5"])

        try:
            euclidean_distance(x, y)
            assert False

        except TypeError:
            assert True

    def test_invalid_shape_mahalanobis_distance(self):
        """
        Test invalid shape for mahalanobis distance method
        :return:
        """

        x = np.array([0.1, 0.2])
        y = np.array([5])

        try:
            mahalanobis_distance(x, y)
            assert False

        except AttributeError:
            assert True

    def test_invalid_shape_mse_distance(self):
        """
        Test invalid shape for mse distance method
        :return:
        """

        x = np.array([0.1, 0.2])
        y = np.array([5])

        try:
            mse_distance(x, y)
            assert False

        except ValueError:
            assert True

    def test_valid_anomaly_score(self):
        """
        Test valid anomaly_score method
        :return:
        """

        x = np.array([2.0, 3.0])
        y = np.array([1.0, 2.0])

        self.assertEqual(anomaly_score(x, y, "MSE"), 1.0)

    def test_invalid_similarity_function_for_anomaly_score(self):
        """
        Test invalid similarity function for anomaly score method
        :return:
        """

        x = np.array([2.0, 3.0])
        y = np.array([1.0, 2.0])

        try:
            self.assertEqual(anomaly_score(x, y, "Invalid function"), 1.0)
            assert False

        except AssertionError:
            assert True

    def test_invalid_shape_for_anomaly_score(self):
        """
        Test invalid shape for anomaly score method
        :return:
        """

        x = np.array([3.0])
        y = np.array([1.0, 2.0])

        try:
            self.assertEqual(anomaly_score(x, y, "Euclidean distance"), 1.0)
            assert False

        except AssertionError:
            assert True

    def test_empty_anomaly_score(self):
        """
        Test anomaly score method for empty data
        :return:
        """

        x = np.array([])
        y = np.array([])

        try:
            self.assertEqual(anomaly_score(x, y, "Mahalanobis distance"), 1.0)
            assert False

        except AssertionError:
            assert True

    def test_valid_anomaly_score_multi(self):
        """
        Test valid anomaly score method
        :return:
        """

        x = np.array([[[1, 2], [1, 3]]])
        y = np.array([[[1, 2], [3, 1]]])

        self.assertEqual(anomaly_score_multi(x, y, "MSE"), 2.0)

    def test_invalid_anomaly_score_multi(self):
        """
        Test invalid anomaly score method
        :return:
        """

        x = np.array([[[1, 3]]])
        y = np.array([[[1, 2], [3, 1]]])

        try:
            self.assertEqual(anomaly_score_multi(x, y, "Euclidean distance"), 2.0)
            assert False

        except AssertionError:
            assert True

    def test_empty_anomaly_score_multi(self):
        """
        Test anomaly score method for wmpty data
        :return:
        """

        x = np.array([[]])
        y = np.array([[]])

        try:
            self.assertEqual(anomaly_score_multi(x, y, "Mahalanobis distance"), 2.0)
            assert False

        except AssertionError:
            assert True

    def test_valid_get_training_data_lstm(self):
        """
        Test valid get training data lstm
        :return:
        """

        list = np.array([[2, 4], [1, 3], [3, 4]])

        desired = np.array([[[2, 4], [1, 3]], [[1, 3], [3, 4]]])

        self.assertEqual(get_training_data_lstm(list, 2).all(), desired.all())

    def test_invalid_window_for_get_training_data_lstm(self):
        """
        Test invalid window for get training data lstm method
        :return:
        """

        list = np.array([[2, 4], [1, 3], [3, 4]])

        desired = np.array([[[]]])

        self.assertEqual(get_training_data_lstm(list, -3).all(), desired.all())

    def test_valid_get_testing_data_lstm(self):
        """
        Test valid get testing data lstm
        :return:
        """

        list = np.array([[2, 4], [1, 3], [3, 4]])

        desired_list = np.array([[[2, 4], [1, 3]], [[1, 3], [3, 4]]])

        labels = np.array([0, 1, 0])

        desired_labels = np.array([[[1, 1]]])

        self.assertEqual(get_testing_data_lstm(list, labels, 2)[0].all(), desired_list.all())
        self.assertEqual(get_testing_data_lstm(list, labels, 2)[1].all(), desired_labels.all())

    def test_invalid_window_for_get_testing_data_lstm(self):
        """
        Test invalid window for get testing data lstm
        :return:
        """

        list = np.array([[2, 4], [1, 3], [3, 4]])

        labels = np.array([0, 1, 0])
        try:
            get_testing_data_lstm(list, labels, -3)
            assert False

        except ValueError:
            assert True

    def test_valid_get_threshold(self):
        """
        Test valid get_threshold
        :return:
        """

        scores = np.array([1, 2, 3])

        percent = 0.95

        self.assertEqual(get_threshold(scores, percent), 2)

    def test_invalid_get_threshold(self):
        """
        Test invalid percent for get_threshold
        :return:
        """

        scores = np.array([1, 2, 3])

        percent = 5.2

        try:
            self.assertEqual(get_threshold(scores, percent), 2)
            assert False

        except AssertionError:
            assert True

    def test_valid_get_attack_boundaries(self):
        """
        Test valid input
        :return:
        """

        df_label = pd.DataFrame(np.array([[0], [0], [1], [1]]))
        expected_output = [2, 3]
        actual_output = get_attack_boundaries(df_label)

        self.assertEqual(expected_output[0], actual_output[0])
        self.assertEqual(expected_output[1], actual_output[1])

    def test_no_spoofing_get_attack_boundaries(self):
        """
        Test invalid input without spoofing
        :return:
        """

        df_label = pd.DataFrame(np.array([[0], [0], [0], [0]]))
        expected_output = [None, 3]
        actual_output = get_attack_boundaries(df_label)

        self.assertEqual(expected_output[0], actual_output[0])
        self.assertEqual(expected_output[1], actual_output[1])

    def test_valid_get_method_scores(self):
        """
        Test valid input
        :return:
        """

        expected_output = [1.0, 0.5, 0.6666666666666666, 0, 1.0]
        tpr, fpr, acc, detection_delay, attack_duration = get_method_scores(prediction=self.prediction,
                                                                            attack_start=self.attack_start,
                                                                            attack_end=self.attack_end,
                                                                            add_window_size=self.add_window_size,
                                                                            window_size=self.window_size)

        self.assertEqual(tpr, expected_output[0])
        self.assertEqual(fpr, expected_output[1])
        self.assertEqual(acc, expected_output[2])
        self.assertEqual(detection_delay, expected_output[3])
        self.assertEqual(attack_duration, expected_output[4])

    def test_invalid_attack_boundary_get_method_scores(self):
        """
        Test invalid attack boundary
        :return:
        """

        try:
            get_method_scores(prediction=self.prediction,
                              attack_start=3,
                              attack_end=1,
                              add_window_size=self.add_window_size,
                              window_size=self.window_size)
            assert False

        except AssertionError:
            assert True

    def test_invalid_prediction_length_get_method_scores(self):
        """
        Test invalid prediction
        :return:
        """

        try:
            get_method_scores(prediction=self.prediction,
                              attack_start=2,
                              attack_end=4,
                              add_window_size=self.add_window_size,
                              window_size=self.window_size)
            assert False

        except AssertionError:
            assert True

    def test_valid_no_attack_get_method_scores(self):
        """
        Test valid  without attack
        :return:
        """

        expected_output = [0, 0, 0.6666666666666666, 1.0, 1.0]
        tpr, fpr, acc, detection_delay, attack_duration = get_method_scores(prediction=[0, 0, 0],
                                                                            attack_start=self.attack_start,
                                                                            attack_end=self.attack_end,
                                                                            add_window_size=self.add_window_size,
                                                                            window_size=self.window_size)

        self.assertEqual(tpr, expected_output[0])
        self.assertEqual(fpr, expected_output[1])
        self.assertEqual(acc, expected_output[2])
        self.assertEqual(detection_delay, expected_output[3])
        self.assertEqual(attack_duration, expected_output[4])
