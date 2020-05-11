'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Tests for models/lstm/lstm_autoencoder
'''

import unittest

from models.lstm.lstm_autoencoder import get_lstm_autoencoder_model


class TestModelsLstmAutoencoder(unittest.TestCase):

    def setUp(self):
        """
        Set up values before test
        :return:
        """

        self.timestmaps = 2
        self.input_features = 2
        self.target_features = 2
        self.encoding_dimenstion = 10
        self.activation = 'softmax'
        self.loss = 'mean_squared_error'
        self.optimizer = 'Adam'

    def test_valid_get_lstm_autoencoder_model(self):
        """
        Test valid input parameters
        :return:
        """

        lstm_model = get_lstm_autoencoder_model(timesteps=self.timestmaps,
                                                input_features=self.input_features,
                                                target_features=self.target_features,
                                                encoding_dimension=self.encoding_dimenstion,
                                                activation=self.activation,
                                                loss=self.loss,
                                                optimizer=self.optimizer)

        self.assertEqual(lstm_model.loss, self.loss)

    def test_invalid_loss_get_lstm_autoencoder_model(self):
        """
        Test invalid input loss function
        :return:
        """

        invalid_loss_function = 'invalid_loss'

        try:
            get_lstm_autoencoder_model(timesteps=self.timestmaps,
                                       input_features=self.input_features,
                                       target_features=self.target_features,
                                       encoding_dimension=self.encoding_dimenstion,
                                       activation=self.activation,
                                       loss=invalid_loss_function,
                                       optimizer=self.optimizer)
            assert False
        except ValueError:
            assert True

    def test_invalid_optimizer_get_lstm_autoencoder_model(self):
        """
        Test invalid input optimizer
        :return:
        """

        invalid_optimizer = 'invalid_opt'

        try:
            get_lstm_autoencoder_model(timesteps=self.timestmaps,
                                       input_features=self.input_features,
                                       target_features=self.target_features,
                                       encoding_dimension=self.encoding_dimenstion,
                                       activation=self.activation,
                                       loss=self.loss,
                                       optimizer=invalid_optimizer)
            assert False
        except ValueError:
            assert True
