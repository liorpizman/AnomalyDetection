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

import os
import random
import shutil

from pandas import DataFrame
from scripts.simulator.shared.index import write_data_frame_to_csv
from utils.helper_methods import create_directories, get_subdirectories


def get_data_columns():
    """
    get the data columns of train set file
    :return:train set file's columns
    """

    return [
        "TestModeType",
        "NameOfBaseStation",
        "NameOfWpt",
        "NameOfDrone",
        "IpOfHome",
        "Longitude",
        "Latitude",
        "Height",
        "Frequency",
        "Velocity",
        "ParabolicAntennaTypeIndex",
        "OmniAntennaTypeIndex",
        "IpOfDrone",
        "IsNoLinkActiveInWpt",
        "IsSpoofingActivatedInWpt",
        "ForwardType",
        "NameOfSpoofer"
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


def create_train_file(num_of_way_points,
                      velocity_state,
                      height_state,
                      directory_path,
                      file_name):
    """
    creation of basic route file
    :param num_of_way_points: num of way points in route
    :param velocity_state: can be 'Up' , 'Down' or 'Stable'
    :param height_state: can be 'Up' , 'Down' or 'Stable'
    :param directory_path: file location
    :param file_name: route name
    :return: train file for simulator
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


def get_random_state():
    """
    creation of random flight state
    :return: 'Up' , 'Down' or 'Stable'
    """

    state = random.randint(1, 3)
    switcher = {
        1: 'Up',
        2: 'Down',
        3: 'Stable'
    }
    return switcher.get(state, 'Stable')


def create_train_set(source_folder, files_amount, velocity_state, height_state):
    """
    creation of train routes files
    :param source_folder: source path
    :param files_amount: files amount
    :param velocity_state: 'Up' , 'Down' or 'Stable'
    :param height_state: 'Up' , 'Down' or 'Stable'
    :return:
    """

    for i in range(files_amount):
        route_name = "route_" + str(i)
        create_directories(f'{source_folder}/{route_name}')
        create_train_file(num_of_way_points=random.randint(10, 20),
                          velocity_state=velocity_state,
                          height_state=height_state,
                          directory_path=f'{source_folder}/{route_name}',
                          file_name=route_name)


def move_file_to_target_path(source_directory, target_directory,
                             source_file_name, target_file_name):
    """
    move files from source path to target path
    :param source_directory: source path
    :param target_directory: target path
    :param source_file_name: source file name
    :param target_file_name: target file name
    :return:
    """

    shutil.move(f'{source_directory}/{source_file_name}', f'{target_directory}/{target_file_name}')


def get_sensors_file(path):
    """
    get sensors file which is exist in a current path
    :param path: input path
    :return: return sensors file name
    """

    for file in os.listdir(path):
        if file.endswith("_LATEST.csv"):
            return file

    return ""


def move_train_files_to_target_path(source_directory, target_directory, train_folder_name):
    """
    move train files from source path to target path
    :param source_directory: source path
    :param target_directory: target path
    :param train_folder_name: the name of the route
    :return:
    """

    flight_files = get_subdirectories(source_directory)

    for index, route in enumerate(flight_files):
        new_route_name = train_folder_name + "Route_" + str(index)
        current_directory = os.path.join(source_directory, route)
        sensors_file = get_sensors_file(current_directory)
        create_directories(f'{target_directory}/{new_route_name}')
        move_file_to_target_path(current_directory, f'{target_directory}/{new_route_name}',
                                 sensors_file, "without_anom.csv")


# source_folder = "C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\simulator_final_train_set\\latest_files"
# logs_folder = "C:\\Users\\Lior\\Desktop\\Simulator Versions\\Simulator\\Logs"
# target_folder = "C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\simulator_final_train_set\\train_set"

# source_folder = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\simulator_data_set\\train_2"
# logs_folder = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\Simulator\\Logs"
# target_folder = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\simulator_data_set\\suitable_set\\train"

files_amount = 2

# velocity_state = ['Up' , 'Down' ,'Stable' ]
# height_state = ['Up' , 'Down' ,'Stable' ]

velocity_state = 'None'  # Note: change it just for run the script and replace it back to None!
height_state = 'None'  # Note: change it just for run the script and replace it back to None!

flight_suffix = '\\[Velocity = {0}] [Height = {1}]'.format(velocity_state, height_state)

# Note: do not run this function with the same params - it will override the data there!!!
# create_train_set(source_folder=source_folder + flight_suffix,
#                  files_amount=files_amount,
#                  velocity_state=velocity_state,
#                  height_state=height_state)

velocity_arg = 'Up'  # Note: change it just for run the script and replace it back to None!
height_arg = 'Up'  # Note: change it just for run the script and replace it back to None!

train_folder_name = '[Velocity = {0}]_[Height = {1}]_'.format(velocity_arg, height_arg)

# You should delete logs folder before each run!
# move_train_files_to_target_path(source_directory=logs_folder,
#                                 target_directory=target_folder,
#                                 train_folder_name=train_folder_name)
