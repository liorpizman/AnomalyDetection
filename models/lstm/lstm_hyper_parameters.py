#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
LSTM hyper parameters to reach best LSTM model
'''
import win32api


class lstm_hyper_parameters:
    """
    A Class used to hyper LSTM parameters
    """

    # number of training samples to use for encoding and decoding each time for the LSTM
    LSTM_WINDOW_SIZE = None

    # encoding dimension for LSTM autoencoder
    LSTM_ENCODING_DIMENSION = None

    # determine threshold for classification from this percent of training set that had lower errors
    # (e.g get the threshold error in which 99% of training set had lower values than)
    LSTM_THRESHOLD_FROM_TRAINING_PERCENT = None

    LSTM_ACTIVATION = None

    LSTM_LOSS = None

    LSTM_OPTIMIZER = None

    LSTM_EPOCHS = None

    """
    Attributes
    ----------
    LSTM_WINDOW_SIZE                              : int

    LSTM_ENCODING_DIMENSION                       : int 

    LSTM_THRESHOLD_FROM_TRAINING_PERCENT          : float

    LSTM_ACTIVATION                               : str

    LSTM_LOSS                                     : str

    LSTM_OPTIMIZER                                : str

    LSTM_EPOCHS                                   : int 
    
    Methods
    -------
    set_window_size(window_size)
            Description | Set new value for window size

    remove_window_size(window_size)
            Description | Remove current value of window size
            
    get_window_size()
            Description | Get current value of window size
            
    set_encoding_dimension(encoding_dimension)
            Description | Set new value for encoding dimension 
            
    remove_encoding_dimension(encoding_dimension)
            Description | Remove current value of encoding dimension 
            
    get_encoding_dimension()
            Description | Get current value of encoding dimension  
            
    set_activation(activation)
            Description | Set new value for activation function
            
    remove_activation(activation)
            Description | Remove current value of activation function

    get_activation()
            Description | Get current value of activation function

    set_loss(loss)
            Description | Set new value for loss function
            
    remove_loss(loss)
            Description | Remove current value of loss function
            
    set_epochs()
            Description | Set new value for the epochs variable
            
    remove_epochs(epochs)
            Description | Remove current value of epochs variable
            
    get_epochs()
            Description | Get current value of epochs variable
            
    get_loss()
            Description |  Get current value of loss function
            
    set_optimizer(optimizer)
            Description | Set new value for the optimizer
            
    remove_optimizer(optimizer)
            Description | Remove current value of optimizer
            
    set_threshold(threshold)
            Description | Set new value for the threshold
            
    remove_threshold(threshold)
            Description | Remove current value of threshold

    get_threshold()
            Description | Get current value of threshold

    """

    # Window size parameter
    @staticmethod
    def set_window_size(window_size):
        lstm_hyper_parameters.LSTM_WINDOW_SIZE = int(window_size)

    @staticmethod
    def remove_window_size(window_size):
        lstm_hyper_parameters.LSTM_WINDOW_SIZE = None

    @staticmethod
    def get_window_size():
        return lstm_hyper_parameters.LSTM_WINDOW_SIZE

    # Encoding dimension parameter
    @staticmethod
    def set_encoding_dimension(encoding_dimension):
        lstm_hyper_parameters.LSTM_ENCODING_DIMENSION = int(encoding_dimension)

    @staticmethod
    def remove_encoding_dimension(encoding_dimension):
        lstm_hyper_parameters.LSTM_ENCODING_DIMENSION = None

    @staticmethod
    def get_encoding_dimension():
        return lstm_hyper_parameters.LSTM_ENCODING_DIMENSION

    # Activation parameter
    @staticmethod
    def set_activation(activation):
        lstm_hyper_parameters.LSTM_ACTIVATION = activation

    @staticmethod
    def remove_activation(activation):
        lstm_hyper_parameters.LSTM_ACTIVATION = None

    @staticmethod
    def get_activation():
        return lstm_hyper_parameters.LSTM_ACTIVATION

    # Loss parameter
    @staticmethod
    def set_loss(loss):
        lstm_hyper_parameters.LSTM_LOSS = loss

    @staticmethod
    def remove_loss(loss):
        lstm_hyper_parameters.LSTM_LOSS = None

    @staticmethod
    def get_loss():
        return lstm_hyper_parameters.LSTM_LOSS

    # Epochs parameter
    @staticmethod
    def set_epochs(epochs):
        lstm_hyper_parameters.LSTM_EPOCHS = int(epochs)

    @staticmethod
    def remove_epochs(epochs):
        lstm_hyper_parameters.LSTM_EPOCHS = None

    @staticmethod
    def get_epochs():
        return lstm_hyper_parameters.LSTM_EPOCHS

    # Optimizer parameter
    @staticmethod
    def set_optimizer(optimizer):
        lstm_hyper_parameters.LSTM_OPTIMIZER = optimizer

    @staticmethod
    def remove_optimizer(optimizer):
        lstm_hyper_parameters.LSTM_OPTIMIZER = None

    @staticmethod
    def get_optimizer():
        return lstm_hyper_parameters.LSTM_OPTIMIZER

    # Threshold parameter
    @staticmethod
    def set_threshold(threshold):
        lstm_hyper_parameters.LSTM_THRESHOLD_FROM_TRAINING_PERCENT = float(threshold)

    @staticmethod
    def remove_threshold(threshold):
        lstm_hyper_parameters.LSTM_THRESHOLD_FROM_TRAINING_PERCENT = None

    @staticmethod
    def get_threshold():
        return lstm_hyper_parameters.LSTM_THRESHOLD_FROM_TRAINING_PERCENT
