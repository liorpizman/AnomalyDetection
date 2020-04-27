#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Project Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
Source code: https://github.com/esvhd/TimeSeriesRegressor
File created by: esvhd
---
Utils for Time Series Estimator wrapper for machine learning models
'''

import numpy as np


def safe_shape(array, i):
    """
    Safe shape function to return default value in case of positive index
    :param array: input array
    :param i: index to shape
    :return: valid index when positive i
    """

    try:
        return array.shape[i]
    except IndexError:
        if i > 0:
            return 1
        else:
            raise IndexError


def mse(X1, X2, multi_output='raw_values'):
    """
    Calculate MSE between two data frames
    :param X1: first data frame
    :param X2: second data frame
    :param multi_output: method to return
    :return: calculated MSE
    """

    if multi_output == 'raw_values':
        return np.mean((X1 - X2) ** 2, axis=0) ** .5
    if multi_output == 'uniform_average':
        return np.mean(np.mean((X1 - X2) ** 2, axis=0) ** .5)
