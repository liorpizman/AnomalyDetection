#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Random forest hyper parameters to reach best Random forest model
'''


class random_forest_hyper_parameters:
    """
    A Class used to hyper Random forest parameters
    """

    WINDOW_SIZE = None
    N_ESTIMATORS = None
    CRITERION = None
    MAX_FEATURES = None
    RANDOM_STATE = None
    THRESHOLD = None

    DEFAULT_THRESHOLD = 0.99
    """
    Attributes
    ----------
    WINDOW_SIZE                                : int
    
    N_ESTIMATORS                               : int

    CRITERION                                  : str 

    MAX_FEATURES                               : str

    RANDOM_STATE                               : int

    THRESHOLD                                  : float

    Methods
    -------
    set_window_size(window_size)
        Description | Set new value for window size

    remove_window_size(window_size)
            Description | Remove current value of window size
            
    get_window_size()
            Description | Get current value of window size
            
    set_n_estimators(n_estimators)
            Description | Set new value for n estimators

    remove_n_estimators(n_estimators)
            Description | Remove current value of n estimators
            
    get_n_estimators()
            Description | Get current value of n estimators

    set_criterion(criterion)
            Description | Set new value for criterion

    remove_criterion(criterion)
            Description | Remove current value of criterion
            
    get_criterion()
            Description | Get current value of criterion
            
    set_max_features(max_features)
            Description | Set new value for max features

    remove_max_features(max_features)
            Description | Remove current value of max features
            
    get_max_features()
            Description | Get current value of max features
            
    set_random_state(random_state)
            Description | Set new value for random state

    remove_random_state(random_state)
            Description | Remove current value of random state
            
    get_random_state()
            Description | Get current value of random state
            
    set_threshold(threshold)
            Description | Set new value for threshold

    remove_threshold(threshold)
            Description | Remove current value of threshold
            
    get_threshold()
            Description | Get current value of threshold

    """

    # Window size parameter
    @staticmethod
    def set_window_size(window_size):
        random_forest_hyper_parameters.WINDOW_SIZE = int(window_size)

    @staticmethod
    def remove_window_size(window_size):
        random_forest_hyper_parameters.WINDOW_SIZE = None

    @staticmethod
    def get_window_size():
        return random_forest_hyper_parameters.WINDOW_SIZE

    # N estimators parameter
    @staticmethod
    def set_n_estimators(n_estimators):
        random_forest_hyper_parameters.N_ESTIMATORS = int(n_estimators)

    @staticmethod
    def remove_n_estimators():
        random_forest_hyper_parameters.N_ESTIMATORS = None

    @staticmethod
    def get_n_estimators():
        return random_forest_hyper_parameters.N_ESTIMATORS

    # Criterion parameter
    @staticmethod
    def set_criterion(criterion):
        random_forest_hyper_parameters.CRITERION = criterion

    @staticmethod
    def remove_criterion():
        random_forest_hyper_parameters.CRITERION = None

    @staticmethod
    def get_criterion():
        return random_forest_hyper_parameters.CRITERION

    # Max features parameter
    @staticmethod
    def set_max_features(max_features):
        random_forest_hyper_parameters.MAX_FEATURES = max_features

    @staticmethod
    def remove_max_features():
        random_forest_hyper_parameters.MAX_FEATURES = None

    @staticmethod
    def get_max_features():
        return random_forest_hyper_parameters.MAX_FEATURES

    # Random state parameter
    @staticmethod
    def set_random_state(random_state):
        random_forest_hyper_parameters.RANDOM_STATE = int(random_state)

    @staticmethod
    def remove_random_state():
        random_forest_hyper_parameters.RANDOM_STATE = None

    @staticmethod
    def get_random_state():
        return random_forest_hyper_parameters.RANDOM_STATE

    # Threshold parameter
    @staticmethod
    def set_threshold(threshold):
        random_forest_hyper_parameters.THRESHOLD = float(threshold)

    @staticmethod
    def remove_threshold():
        random_forest_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return random_forest_hyper_parameters.THRESHOLD
