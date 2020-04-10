#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
KNN hyper parameters to reach best KNN model
'''


class knn_hyper_parameters:
    """
    A Class used to hyper KNN parameters
    """

    NEIGHBORS_NUMBER = None
    WEIGHTS = None
    ALGORITHM = None
    THRESHOLD = None

    """
    Attributes
    ----------
    NEIGHBORS_NUMBER                        : int

    WEIGHTS                                 : str 

    ALGORITHM                               : str

    THRESHOLD                               : float

    Methods
    -------
    set_n_neighbors()
            Description | Set new value for n neighbors 

    remove_n_neighbors()
            Description | Remove current value of n neighbors 
            
    get_n_neighbors()
            Description | Get current value of n neighbors 
            
    set_weights()
            Description | Set new value for weights variable

    remove_weights()
            Description | Remove current value of weights variable

    get_weights()
            Description | Get current value of weights variable
            
    set_algorithm()
            Description | Set new value for algorithm
            
    remove_algorithm()
            Description | Remove current value of algorithm
            
    get_algorithm()
            Description | Get current value of algorithm

    set_threshold()
            Description | Set new value for threshold
            
    remove_threshold()
            Description | Remove current value of threshold
            
    get_threshold()
            Description | Get current value of threshold
            
    """

    @staticmethod
    def set_n_neighbors(n_neighbors):
        knn_hyper_parameters.NEIGHBORS_NUMBER = int(n_neighbors)

    @staticmethod
    def remove_n_neighbors():
        knn_hyper_parameters.NEIGHBORS_NUMBER = None

    @staticmethod
    def get_n_neighbors():
        return knn_hyper_parameters.NEIGHBORS_NUMBER

    @staticmethod
    def set_weights(weights):
        knn_hyper_parameters.WEIGHTS = weights

    @staticmethod
    def remove_weights():
        knn_hyper_parameters.WEIGHTS = None

    @staticmethod
    def get_weights():
        return knn_hyper_parameters.WEIGHTS

    @staticmethod
    def set_algorithm(algorithm):
        knn_hyper_parameters.ALGORITHM = algorithm

    @staticmethod
    def remove_algorithm():
        knn_hyper_parameters.ALGORITHM = None

    @staticmethod
    def get_algorithm():
        return knn_hyper_parameters.ALGORITHM

    @staticmethod
    def set_threshold(threshold):
        knn_hyper_parameters.THRESHOLD = float(threshold)

    @staticmethod
    def remove_threshold():
        knn_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return knn_hyper_parameters.THRESHOLD
