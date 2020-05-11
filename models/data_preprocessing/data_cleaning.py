#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Fill in missing values, smooth noisy data, identify or remove outliers, and resolve inconsistencies
'''

import pandas as pd

from utils.constants import COLUMNS_TO_REMOVE


def clean_data(data):
    """
    Clean the data by different steps as part of data pre-processing
    :param data: input data frame
    :return: cleaned data frame
    """

    # Export constant columns which should be dropped
    to_drop = COLUMNS_TO_REMOVE

    # Step 1 : drop unnecessary columns
    dropped_columns_data = drop_columns(data, to_drop)

    # Step 2 : fill in missing values
    removed_na_data = remove_na(dropped_columns_data)

    return removed_na_data


def drop_columns(data, to_drop):
    """
    Drop unnecessary columns
    :param data: input data frame
    :param to_drop: constant columns which should be dropped
    :return: data frame without the columns in the input
    """

    # If 'ignore', suppress error and only existing labels are dropped.
    # errors : {'ignore', 'raise'}, default 'raise'
    transformed_data = data.drop(columns=to_drop, inplace=False, errors='ignore')

    return transformed_data


def remove_na(data):
    """
    Fill in missing values
    :param data: input data frame
    :return: data frame without na values
    """

    columns = list(data)
    transformed_data = pd.DataFrame(columns=columns)

    # Fill NAN with last valid value
    # 'ffill' stands for 'forward fill' and will propagate last valid observation forward
    for column_name in columns:
        transformed_data[column_name] = data[column_name].fillna(method='ffill', inplace=False)

    return transformed_data
