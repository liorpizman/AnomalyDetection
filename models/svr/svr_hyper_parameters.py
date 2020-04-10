#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
SVR hyper parameters to reach best SVR model
'''


class svr_hyper_parameters:
    """
    A Class used to hyper SVR parameters
    """

    KERNEL = None
    GAMMA = None
    EPSILON = None
    THRESHOLD = None
    DEFAULT_EPSILON = 0.1
    DEFAULT_THRESHOLD = 0.99

    """
    Attributes
    ----------
    KERNEL                                  : str

    GAMMA                                   : str 

    EPSILON                                 : float

    THRESHOLD                               : float


    Methods
    -------
    set_kernel()
            Description | Set new value for kernel

    remove_kernel()
            Description | Remove current value of kernel

    get_kernel()
            Description | Get current value of kernel

    set_gamma()
            Description | Set new value for gamma

    remove_gamma()
            Description | Remove current value of gamma

    get_gamma()
            Description | Get current value of gamma

    set_epsilon()
            Description | Set new value for epsilon

    remove_epsilon()
            Description | Remove current value of epsilon

    get_epsilon()
            Description | Get current value of epsilon

    set_threshold()
            Description | Set new value for threshold

    remove_threshold()
            Description | Remove current value of threshold

    get_threshold()
            Description | Get current value of threshold

    """

    @staticmethod
    def set_kernel(kernel):
        svr_hyper_parameters.KERNEL = kernel

    @staticmethod
    def remove_kernel():
        svr_hyper_parameters.KERNEL = None

    @staticmethod
    def get_kernel():
        return svr_hyper_parameters.KERNEL

    @staticmethod
    def set_gamma(gamma):
        svr_hyper_parameters.GAMMA = gamma

    @staticmethod
    def remove_gamma():
        svr_hyper_parameters.GAMMA = None

    @staticmethod
    def get_gamma():
        return svr_hyper_parameters.GAMMA

    @staticmethod
    def set_epsilon(epsilon):
        try:
            svr_hyper_parameters.EPSILON = float(epsilon)
        except:
            svr_hyper_parameters.EPSILON = svr_hyper_parameters.DEFAULT_EPSILON

    @staticmethod
    def remove_epsilon():
        svr_hyper_parameters.EPSILON = None

    @staticmethod
    def get_epsilon():
        return svr_hyper_parameters.EPSILON

    @staticmethod
    def set_threshold(threshold):
        try:
            svr_hyper_parameters.THRESHOLD = float(threshold)
        except:
            svr_hyper_parameters.THRESHOLD = svr_hyper_parameters.DEFAULT_THRESHOLD

    @staticmethod
    def remove_threshold():
        svr_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return svr_hyper_parameters.THRESHOLD
