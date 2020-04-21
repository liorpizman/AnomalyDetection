#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Normalization and aggregation methods as part of data pre-processing
'''

from sklearn.preprocessing import PowerTransformer, MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler, \
    QuantileTransformer


def normalize_data(data, scaler):
    """
    Data normalization by using a chosen scaler
    :param data: input data frame
    :param scaler: string for chosen scaler
    :return: scaled data frame
    """

    # Switch between different scalers
    # Helpful source: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    switcher = {
        "max_abs": MaxAbsScaler(),
        "standard": StandardScaler(),
        "min_max": MinMaxScaler(),
        "robust_scaler": RobustScaler(),
        "quantile_transformer": QuantileTransformer(output_distribution='normal'),
        "power_transform": PowerTransformer(method='yeo-johnson')
    }

    scaler = switcher.get(scaler, StandardScaler())

    return scaler.fit_transform(data), scaler
