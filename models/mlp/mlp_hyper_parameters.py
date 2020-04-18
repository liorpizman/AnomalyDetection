#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
MLP hyper parameters to reach best MLP model
'''


class mlp_hyper_parameters:
    """
    A Class used to hyper MLP parameters
    """

    HIDDEN_LAYER_SIZES = None
    ACTIVATION = None
    SOLVER = None
    ALPHA = None
    RANDOM_STATE = None
    THRESHOLD = None

    DEFAULT_THRESHOLD = 0.99

    """
    Attributes
    ----------
    HIDDEN_LAYER_SIZES                     : tuple
    
    ACTIVATION                             : str
    
    SOLVER                                 : str
    
    ALPHA                                  : float
    
    RANDOM_STATE                           : int
    
    THRESHOLD                              : float

    Methods
    -------
    set_hidden_layer_sizes()
            Description | Set new value for hidden layer sizes

    remove_hidden_layer_sizes()
            Description | Remove current value of hidden layer sizes

    get_hidden_layer_sizes()
            Description | Get current value of hidden layer sizes
            
    set_activation()
            Description | Set new value for activation

    remove_activation()
            Description | Remove current value of activation

    get_activation()
            Description | Get current value of activation

    set_solver()
            Description | Set new value for solver

    remove_solver()
            Description | Remove current value of solver

    get_solver()
            Description | Get current value of solver
            
    set_alpha()
            Description | Set new value for alpha

    remove_alpha()
            Description | Remove current value of alpha

    get_alpha()
            Description | Get current value of alpha
            
    set_random_state()
            Description | Set new value for random state

    remove_random_state()
            Description | Remove current value of random state

    get_random_state()
            Description | Get current value of random state
    
    set_threshold()
            Description | Set new value for threshold

    remove_threshold()
            Description | Remove current value of threshold

    get_threshold()
            Description | Get current value of threshold
    """

    # Hidden layer sizes parameter
    @staticmethod
    def set_hidden_layer_sizes(hidden_layer_sizes):
        mlp_hyper_parameters.HIDDEN_LAYER_SIZES = eval(hidden_layer_sizes)

    @staticmethod
    def remove_hidden_layer_sizes():
        mlp_hyper_parameters.HIDDEN_LAYER_SIZES = None

    @staticmethod
    def get_hidden_layer_sizes():
        return mlp_hyper_parameters.HIDDEN_LAYER_SIZES

    # Activation parameter
    @staticmethod
    def set_activation(activation):
        mlp_hyper_parameters.ACTIVATION = activation

    @staticmethod
    def remove_activation():
        mlp_hyper_parameters.ACTIVATION = None

    @staticmethod
    def get_activation():
        return mlp_hyper_parameters.ACTIVATION

    # Solver sizes parameter
    @staticmethod
    def set_solver(solver):
        mlp_hyper_parameters.SOLVER = solver

    @staticmethod
    def remove_solver():
        mlp_hyper_parameters.SOLVER = None

    @staticmethod
    def get_solver():
        return mlp_hyper_parameters.SOLVER

    # Alpha sizes parameter
    @staticmethod
    def set_alpha(alpha):
        mlp_hyper_parameters.ALPHA = float(alpha)

    @staticmethod
    def remove_alpha():
        mlp_hyper_parameters.ALPHA = None

    @staticmethod
    def get_alpha():
        return mlp_hyper_parameters.ALPHA

    # Random state sizes parameter
    @staticmethod
    def set_random_state(random_state):
        mlp_hyper_parameters.RANDOM_STATE = int(random_state)

    @staticmethod
    def remove_random_state():
        mlp_hyper_parameters.RANDOM_STATE = None

    @staticmethod
    def get_random_state():
        return mlp_hyper_parameters.RANDOM_STATE

    # Threshold parameter
    @staticmethod
    def set_threshold(threshold):
        try:
            mlp_hyper_parameters.THRESHOLD = float(threshold)
        except:
            mlp_hyper_parameters.THRESHOLD = mlp_hyper_parameters.DEFAULT_THRESHOLD

    @staticmethod
    def remove_threshold():
        mlp_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return mlp_hyper_parameters.THRESHOLD
