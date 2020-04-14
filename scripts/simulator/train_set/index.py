#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Script to create train set data for different routes by the simulator
'''

import random

from pandas import DataFrame

from scripts.simulator.shared.index import write_data_frame_to_csv


def get_data_columns():
    """
    get the data columns of train set file
    :return:train set file's columns
    """

    return [
        "TestModeType", "NameOfBaseStation", "NameOfWpt",
        "NameOfDrone", "IpOfHome", "Longitude",
        "Latitude", "Height", "Frequency", "Velocity",
        "ParabolicAntennaTypeIndex", "OmniAntennaTypeIndex",
        "IpOfDrone", "IsNoLinkActiveInWpt", "IsSpoofingActivatedInWpt",
        "ForwardType", "NameOfSpoofer"
    ]


def test_mode_type_column_creation(num_of_way_points):
    """
    create TestModeType column
    :param num_of_way_points: num of way points in route
    :return:TestModeType column's data
    """

    return ["TA" if index == num_of_way_points + 2
            else "-"
            for index in range(num_of_way_points + 4)
            ]


def name_of_base_station_column_creation(num_of_way_points):
    """
    create NameOfBaseStation column
    :param num_of_way_points: num of way points in route
    :return:NameOfBaseStation column's data
    """

    return ["BaseStation #1" if index == 0
            else "-"
            for index in range(num_of_way_points + 4)
            ]


def name_of_wpt_column_creation(num_of_way_points):
    """
    create NameOfWpt column
    :param num_of_way_points: num of way points in route
    :return:NameOfWpt column's data
    """

    return ["wayPoint{0}".format(index) if 1 <= index <= num_of_way_points
            else "-"
            for index in range(num_of_way_points + 4)]


def name_of_drone_column_creation(num_of_way_points):
    """
    create NameOfDrone column
    :param num_of_way_points: num of way points in route
    :return:NameOfDrone column's data
    """

    return ["Drone #1" if index == num_of_way_points + 1
            else "-"
            for index in range(num_of_way_points + 4)]


def ip_of_home_column_creation(num_of_way_points):
    """
    create IpOfHome column
    :param num_of_way_points: num of way points in route
    :return:IpOfHome column's data
    """

    return ["127.0.0.1" if index == 0
            else "-"
            for index in range(num_of_way_points + 4)]


def longitude_column_creation(num_of_way_points):
    """
    create Longitude column
    :param num_of_way_points: num of way points in route
    :return:Longitude column's data
    """

    return [str(random.uniform(34.660001, 34.959999)) if index <= num_of_way_points
            else "-"
            for index in range(num_of_way_points + 4)]


def latitude_column_creation(num_of_way_points):
    """
    create Latitude column
    :param num_of_way_points: num of way points in route
    :return:Latitude column's data
    """

    return [str(random.uniform(31.888881, 32.009001)) if index <= num_of_way_points
            else "-"
            for index in range(num_of_way_points + 4)]


def height_column_creation(num_of_way_points, state):
    """
    create Height column
    :param num_of_way_points: num of way points in route
    :param state: can be 'Up' , 'Down' or 'Stable'
    :return:Height column's data
    """

    if state == "Stable":
        factor = 0
        last_height = random.uniform(10, 30)
    elif state == "Down":
        factor = -1
        last_height = random.uniform(70, 80)
    else:  # state == "Up"
        factor = 1
        last_height = random.uniform(10, 20)

    column = []

    for index in range(num_of_way_points + 4):

        if index == 0:
            column.append("0")
        elif 1 <= index <= num_of_way_points:
            base = random.randint(1, 3)
            column.append(str(last_height))
            last_height = last_height + factor * base
        else:
            column.append("-")

    return column


def frequency_column_creation(num_of_way_points):
    """
    create Frequency column
    :param num_of_way_points: num of way points in route
    :return:Frequency column's data
    """

    return ["2510" if index == 0
            else "-"
            for index in range(num_of_way_points + 4)]


def velocity_column_creation(num_of_way_points, state):
    """
    create Velocity column
    :param num_of_way_points: num of way points in route
    :param state: can be 'Up' , 'Down' or 'Stable'
    :return:Velocity column's data
    """

    if state == "Stable":
        factor = 0
        last_velocity = random.uniform(10, 30)
    elif state == "Down":
        factor = -1
        last_velocity = random.uniform(40, 50)
    else:  # state == "Up"
        factor = 1
        last_velocity = random.uniform(10, 20)

    column = []

    for index in range(num_of_way_points + 4):

        if 1 <= index <= num_of_way_points:
            base = random.randint(1, 2)
            column.append(str(last_velocity))
            last_velocity = last_velocity + factor * base
        else:
            column.append("-")

    return column


def parabolic_antenna_column_creation(num_of_way_points):
    """
    create ParabolicAntennaTypeIndex column
    :param num_of_way_points: num of way points in route
    :return:ParabolicAntennaTypeIndex column's data
    """

    return ["0" if index == 0
            else "-"
            for index in range(num_of_way_points + 4)]


def omni_antenna_column_creation(num_of_way_points):
    """
    create OmniAntennaTypeIndex column
    :param num_of_way_points: num of way points in route
    :return:OmniAntennaTypeIndex column's data
    """

    return ["0" if index == 0
            else "-"
            for index in range(num_of_way_points + 4)]


def ip_of_drone_column_creation(num_of_way_points):
    """
    create IpOfDrone column
    :param num_of_way_points: num of way points in route
    :return:IpOfDrone column's data
    """

    return ["127.0.0.2" if index == num_of_way_points + 1
            else "-"
            for index in range(num_of_way_points + 4)]


def no_link_column_creation(num_of_way_points):
    """
    create IsNoLinkActiveInWpt column
    :param num_of_way_points: num of way points in route
    :return:IsNoLinkActiveInWpt column's data
    """

    return ["FALSE" if 1 <= index <= num_of_way_points
            else "-"
            for index in range(num_of_way_points + 4)]


def spoofing_activated_column_creation(num_of_way_points):
    """
    create IsSpoofingActivatedInWpt column
    :param num_of_way_points: num of way points in route
    :return:IsSpoofingActivatedInWpt column's data
    """

    return ["FALSE" if 1 <= index <= num_of_way_points
            else "-"
            for index in range(num_of_way_points + 4)]


def forward_type_column_creation(num_of_way_points):
    """
    create ForwardType column
    :param num_of_way_points: num of way points in route
    :return:ForwardType column's data
    """

    return ["max" if index == num_of_way_points + 3
            else "-"
            for index in range(num_of_way_points + 4)]


def name_of_spoofer_column_creation(num_of_way_points):
    """
    create NameOfSpoofer column
    :param num_of_way_points: num of way points in route
    :return:NameOfSpoofer column's data
    """

    return ["-" for index in range(num_of_way_points + 4)]


def create_train_set(num_of_way_points,
                     velocity_state,
                     height_state,
                     directory_path,
                     file_name):
    """
    creation of basic route
    :param num_of_way_points: num of way points in route
    :param velocity_state: can be 'Up' , 'Down' ro 'Stable'
    :param height_state: can be 'Up' , 'Down' ro 'Stable'
    :param directory_path: file location
    :param file_name: route name
    :return:
    """

    df = DataFrame(columns=get_data_columns())
    df["TestModeType"] = test_mode_type_column_creation(num_of_way_points)
    df["NameOfBaseStation"] = name_of_base_station_column_creation(num_of_way_points)
    df["NameOfWpt"] = name_of_wpt_column_creation(num_of_way_points)
    df["NameOfDrone"] = name_of_drone_column_creation(num_of_way_points)
    df["IpOfHome"] = ip_of_home_column_creation(num_of_way_points)
    df["Longitude"] = longitude_column_creation(num_of_way_points)
    df["Latitude"] = latitude_column_creation(num_of_way_points)
    df["Height"] = height_column_creation(num_of_way_points, height_state)
    df["Frequency"] = frequency_column_creation(num_of_way_points)
    df["Velocity"] = velocity_column_creation(num_of_way_points, velocity_state)
    df["ParabolicAntennaTypeIndex"] = parabolic_antenna_column_creation(num_of_way_points)
    df["OmniAntennaTypeIndex"] = omni_antenna_column_creation(num_of_way_points)
    df["IpOfDrone"] = ip_of_drone_column_creation(num_of_way_points)
    df["IsNoLinkActiveInWpt"] = no_link_column_creation(num_of_way_points)
    df["IsSpoofingActivatedInWpt"] = spoofing_activated_column_creation(num_of_way_points)
    df["ForwardType"] = forward_type_column_creation(num_of_way_points)
    df["NameOfSpoofer"] = name_of_spoofer_column_creation(num_of_way_points)

    write_data_frame_to_csv(directory_path=directory_path,
                            file_name=file_name,
                            data_frame=df)


create_train_set(num_of_way_points=7,
                 velocity_state="Up",
                 height_state="Down",
                 directory_path="C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\simulator_data_set\\train_set",
                 file_name="train_0")
