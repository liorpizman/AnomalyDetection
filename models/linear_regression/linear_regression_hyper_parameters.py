#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Linear Regression hyper parameters to reach best Linear Regression model
'''


class linear_regression_hyper_parameters:
    """
    A Class used to hyper Linear Regression parameters
    """

    FIT_INTERCEPT = None

    THRESHOLD = None

    DEFAULT_THRESHOLD = 0.99
    """
    Attributes
    ----------
    FIT_INTERCEPT                           : bool
    
    THRESHOLD                               : float

    Methods
    -------
    set_fit_intercept()
            Description | Set new value for fit_intercept

    remove_fit_intercept()
            Description | Remove current value of fit_intercept

    get_fit_intercept()
            Description | Get current value of fit_intercept
    
    set_threshold()
            Description | Set new value for threshold

    remove_threshold()
            Description | Remove current value of threshold

    get_threshold()
            Description | Get current value of threshold
    """

    @staticmethod
    def set_fit_intercept(kernel):
        linear_regression_hyper_parameters.FIT_INTERCEPT = kernel

    @staticmethod
    def remove_fit_intercept():
        linear_regression_hyper_parameters.FIT_INTERCEPT = None

    @staticmethod
    def get_fit_intercept():
        return linear_regression_hyper_parameters.FIT_INTERCEPT

    @staticmethod
    def set_threshold(threshold):
        try:
            linear_regression_hyper_parameters.THRESHOLD = float(threshold)
        except:
            linear_regression_hyper_parameters.THRESHOLD = linear_regression_hyper_parameters.DEFAULT_THRESHOLD

    @staticmethod
    def remove_threshold():
        linear_regression_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return linear_regression_hyper_parameters.THRESHOLD
