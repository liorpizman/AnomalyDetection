#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Script to detect which features can help us detect anomalies
'''

import pandas as pd

from utils.helper_methods import get_attack_boundaries


def detect_anomaly_features_by_csv_files(without_anomaly_path, spoofed_path):
    """
    Comparison in order to detect which features can give us the best indication for GPS spoofing attack
    :param without_anomaly_path: path of a file without anomaly
    :param spoofed_path: path of a file with anomaly
    :return: list of features which are best indicator for a GPS spoofing attack
    """

    without_anomaly_df = pd.read_csv(without_anomaly_path)
    spoofed_df = pd.read_csv(spoofed_path)  # attack_start: 8039 - attack_end: 8179

    attack_start, attack_end = get_attack_boundaries(spoofed_df['GPS Spoofing'])

    print('window attack : ', attack_end - attack_start)

    drop_end = len(without_anomaly_df)

    without_anomaly_df.drop(without_anomaly_df.loc[0:attack_start - 1].index, inplace=True)
    without_anomaly_df.drop(without_anomaly_df.loc[attack_end + 1:drop_end].index, inplace=True)

    spoofed_df.drop(spoofed_df.loc[0:attack_start - 1].index, inplace=True)

    assert len(without_anomaly_df) == len(spoofed_df)

    return without_anomaly_df == spoofed_df, (without_anomaly_df == spoofed_df).all()


base_directory = 'C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\Scripts\\comparison_features_detection' \
                 '\\[Velocity = Down]_[Height = Down]_Route_0\\'

# Should be inserted by the consumer
_without_anomaly_path = base_directory + 'without_anom.csv'
_spoofed_path = base_directory + 'sensors_0.csv'

# detected_features, detected_columns = detect_anomaly_features_by_csv_files(without_anomaly_path=_without_anomaly_path,
#                                                                            spoofed_path=_spoofed_path)
#
# print(detected_features, detected_columns)
