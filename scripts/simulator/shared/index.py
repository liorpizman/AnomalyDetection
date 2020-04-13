#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Helper for repeatable functions for data set creation
'''


def write_data_frame_to_csv(directory_path, file_name, data_frame):
    """
    write data frame to csv file
    :param directory_path: input path for location of the file
    :param file_name: route name
    :param data_frame: input data
    :return: new csv file created
    """

    data_frame.to_csv(f'{directory_path}/{file_name}.csv', index=False)
