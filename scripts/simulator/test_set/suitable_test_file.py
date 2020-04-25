#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Script to create test set data for different routes by the simulator
'''

import os
import random
import shutil

import pandas as pd
from pandas import DataFrame

from scripts.simulator.shared.index import write_data_frame_to_csv
from utils.helper_methods import create_directories, get_subdirectories


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

    return ["TA" if index == num_of_way_points + 4
            else "-"
            for index in range(num_of_way_points + 6)
            ]


def name_of_base_station_column_creation(num_of_way_points):
    """
    create NameOfBaseStation column
    :param num_of_way_points: num of way points in route
    :return:NameOfBaseStation column's data
    """

    return ["BaseStation #1" if index == 0
            else "-"
            for index in range(num_of_way_points + 6)
            ]


def name_of_wpt_column_creation(num_of_way_points):
    """
    create NameOfWpt column
    :param num_of_way_points: num of way points in route
    :return:NameOfWpt column's data
    """

    return ["wayPoint{0}".format(index) if 1 <= index <= num_of_way_points
            else "-"
            for index in range(num_of_way_points + 6)]


def name_of_drone_column_creation(num_of_way_points):
    """
    create NameOfDrone column
    :param num_of_way_points: num of way points in route
    :return:NameOfDrone column's data
    """

    column = []
    for index in range(num_of_way_points + 6):

        if index == num_of_way_points + 1:
            column.append("Drone #1")
        elif index == num_of_way_points + 2:
            column.append("Drone #2")
        else:
            column.append("-")

    return column


def ip_of_home_column_creation(num_of_way_points):
    """
    create IpOfHome column
    :param num_of_way_points: num of way points in route
    :return:IpOfHome column's data
    """

    return ["127.0.0.1" if index == 0
            else "-"
            for index in range(num_of_way_points + 6)]


def longitude_column_creation(num_of_way_points, source_df):
    """
    create Longitude column
    :param num_of_way_points: num of way points in route
    :param source_df: train file Data Frame
    :return:Longitude column's data
    """

    column_name = "Longitude"
    column = []
    for index in range(num_of_way_points + 6):
        if index <= num_of_way_points:
            column.append(source_df[column_name][index])
        elif index == num_of_way_points + 3:
            column.append(random.uniform(34.660001, 34.959999))
        else:
            column.append("-")

    return column


def latitude_column_creation(num_of_way_points, source_df):
    """
    create Latitude column
    :param num_of_way_points: num of way points in route
    :param source_df: train file Data Frame
    :return:Latitude column's data
    """

    column_name = "Latitude"
    column = []
    for index in range(num_of_way_points + 6):
        if index <= num_of_way_points:
            column.append(source_df[column_name][index])
        elif index == num_of_way_points + 3:
            column.append(random.uniform(31.888881, 32.009001))
        else:
            column.append("-")

    return column


def height_column_creation(num_of_way_points, source_df):
    """
    create Height column
    :param num_of_way_points: num of way points in route
    :param source_df: train file Data Frame
    :return:Height column's data
    """

    column_name = "Height"
    column = []
    for index in range(num_of_way_points + 6):
        if index <= num_of_way_points:
            column.append(source_df[column_name][index])
        elif index == num_of_way_points + 3:
            column.append(random.uniform(30, 30))
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
            for index in range(num_of_way_points + 6)]


def velocity_column_creation(num_of_way_points, source_df):
    """
    create Velocity column
    :param num_of_way_points: num of way points in route
    :param source_df: train file Data Frame
    :return:Velocity column's data
    """

    column_name = "Velocity"
    column = []
    for index in range(num_of_way_points + 6):
        if index <= num_of_way_points:
            column.append(source_df[column_name][index])
        elif index == num_of_way_points + 3:
            column.append(random.uniform(30, 30))
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
            for index in range(num_of_way_points + 6)]


def omni_antenna_column_creation(num_of_way_points):
    """
    create OmniAntennaTypeIndex column
    :param num_of_way_points: num of way points in route
    :return:OmniAntennaTypeIndex column's data
    """

    return ["0" if index == 0
            else "-"
            for index in range(num_of_way_points + 6)]


def ip_of_drone_column_creation(num_of_way_points):
    """
    create IpOfDrone column
    :param num_of_way_points: num of way points in route
    :return:IpOfDrone column's data
    """

    column = []
    for index in range(num_of_way_points + 6):

        if index == num_of_way_points + 1:
            column.append("127.0.0.2")
        elif index == num_of_way_points + 2:
            column.append("127.0.255.2")
        else:
            column.append("-")

    return column


def no_link_column_creation(num_of_way_points):
    """
    create IsNoLinkActiveInWpt column
    :param num_of_way_points: num of way points in route
    :return:IsNoLinkActiveInWpt column's data
    """

    return ["FALSE" if 1 <= index <= num_of_way_points
            else "-"
            for index in range(num_of_way_points + 6)]


def spoofing_activated_column_creation(num_of_way_points,
                                       spoofed_way_point):
    """
    create IsSpoofingActivatedInWpt column
    :param num_of_way_points: num of way points in route
    :param spoofed_way_point: start point of the spoofing attack
    :return:IsSpoofingActivatedInWpt column's data
    """

    column = []
    for index in range(num_of_way_points + 6):

        if index == spoofed_way_point:
            column.append("TRUE")
        elif 1 <= index <= num_of_way_points and index != spoofed_way_point:
            column.append("FALSE")
        else:
            column.append("-")

    return column


def forward_type_column_creation(num_of_way_points):
    """
    create ForwardType column
    :param num_of_way_points: num of way points in route
    :return:ForwardType column's data
    """

    return ["max" if index == num_of_way_points + 5
            else "-"
            for index in range(num_of_way_points + 6)]


def name_of_spoofer_column_creation(num_of_way_points):
    """
    create NameOfSpoofer column
    :param num_of_way_points: num of way points in route
    :return:NameOfSpoofer column's data
    """

    return ["Spoofer" if index == num_of_way_points + 3
            else "-"
            for index in range(num_of_way_points + 6)]


def update_spoofer_values(attack,
                          num_of_way_points,
                          spoofed_way_point,
                          height_source_column,
                          velocity_source_column):
    """
    update spoofer's height and velocity values
    :param attack: can be 'Constant' , 'Height' , 'Velocity' or 'Mixed'
    :param num_of_way_points: num of way points in route
    :param spoofed_way_point: start point of the spoofing attack
    :param height_source_column: way points height values
    :param velocity_source_column: way points velocity values
    :return: height and velocity data
    """

    way_point_index = (spoofed_way_point % num_of_way_points) + 1
    spoofer_index = num_of_way_points + 3
    height_factor = random.choice([-1, 1])
    velocity_factor = random.choice([-1, 1])
    height_base = random.uniform(10, 30)
    velocity_base = random.uniform(10, 30)

    if attack == 'Constant':
        height_source_column[spoofer_index] = float(height_source_column[way_point_index])
        velocity_source_column[spoofer_index] = float(velocity_source_column[way_point_index])

    elif attack == 'Height':
        height_source_column[spoofer_index] = float(height_source_column[way_point_index]) + height_factor * height_base
        velocity_source_column[spoofer_index] = float(velocity_source_column[way_point_index])

    elif attack == 'Velocity':
        height_source_column[spoofer_index] = float(height_source_column[way_point_index])
        velocity_source_column[spoofer_index] = float(velocity_source_column[
                                                          way_point_index]) + velocity_factor * velocity_base

    else:  # attack == 'Mixed'
        height_source_column[spoofer_index] = float(height_source_column[way_point_index]) + height_factor * height_base
        velocity_source_column[spoofer_index] = float(velocity_source_column[
                                                          way_point_index]) + velocity_factor * velocity_base

    return height_source_column, velocity_source_column


def generate_height_and_velocity_columns(num_of_way_points,
                                         source_df,
                                         attack,
                                         spoofed_way_point,
                                         ):
    """
    create Height and Velocity columns
    :param num_of_way_points: num of way points in route
    :param source_df: train file Data Frame
    :param attack: can be 'Constant' , 'Height' , 'Velocity' or 'Mixed'
    :param spoofed_way_point: start point of the spoofing attack
    :return: Height and velocity columns
    """

    height_source_column = height_column_creation(num_of_way_points, source_df)
    velocity_source_column = velocity_column_creation(num_of_way_points, source_df)
    return update_spoofer_values(attack=attack,
                                 num_of_way_points=num_of_way_points,
                                 spoofed_way_point=spoofed_way_point,
                                 height_source_column=height_source_column,
                                 velocity_source_column=velocity_source_column)


def get_random_state():
    """
    creation of random flight state
    :return: 'Up' , 'Down' ro 'Stable'
    """
    state = random.randint(1, 3)
    switcher = {
        1: 'Up',
        2: 'Down',
        3: 'Stable'
    }
    return switcher.get(state, 'Stable')


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


def create_attack_files(source_directory, files_amount, route_name):
    """
    create attack files folders
    :param source_directory: source path
    :param files_amount: files amount
    :param route_name: route file name
    :return:
    """

    for index in range(files_amount):
        create_directories(f'{source_directory}/{route_name}')
        for attack in ['Constant Attack', 'Height Attack', 'Velocity Attack', 'Mixed Attack']:
            create_directories(f'{source_directory}/{route_name}/{attack}')


def create_test_set(num_of_way_points,
                    directory_path,
                    file_name,
                    spoofed_way_point,
                    attack,
                    source_df):
    """
    creation of basic route
    :param num_of_way_points: num of way points in route
    :param directory_path: save location
    :param file_name: route name
    :param spoofed_way_point: start point of the spoofing attack
    :param attack: can be 'Constant' , 'Height' , 'Velocity' or 'Mixed'
    :param source_df: train file Data Frame
    :return:
    """

    df = DataFrame(columns=get_data_columns())
    df["TestModeType"] = test_mode_type_column_creation(num_of_way_points)
    df["NameOfBaseStation"] = name_of_base_station_column_creation(num_of_way_points)
    df["NameOfWpt"] = name_of_wpt_column_creation(num_of_way_points)
    df["NameOfDrone"] = name_of_drone_column_creation(num_of_way_points)
    df["IpOfHome"] = ip_of_home_column_creation(num_of_way_points)
    df["Longitude"] = longitude_column_creation(num_of_way_points, source_df)
    df["Latitude"] = latitude_column_creation(num_of_way_points, source_df)
    df["Frequency"] = frequency_column_creation(num_of_way_points)
    df["ParabolicAntennaTypeIndex"] = parabolic_antenna_column_creation(num_of_way_points)
    df["OmniAntennaTypeIndex"] = omni_antenna_column_creation(num_of_way_points)
    df["IpOfDrone"] = ip_of_drone_column_creation(num_of_way_points)
    df["IsNoLinkActiveInWpt"] = no_link_column_creation(num_of_way_points)
    df["IsSpoofingActivatedInWpt"] = spoofing_activated_column_creation(num_of_way_points, spoofed_way_point)
    df["ForwardType"] = forward_type_column_creation(num_of_way_points)
    df["NameOfSpoofer"] = name_of_spoofer_column_creation(num_of_way_points)

    df["Height"], df["Velocity"] = generate_height_and_velocity_columns(num_of_way_points,
                                                                        source_df,
                                                                        attack,
                                                                        spoofed_way_point,
                                                                        )

    write_data_frame_to_csv(directory_path=directory_path,
                            file_name=file_name,
                            data_frame=df)


def execute_creation(source_directory, target_directory):
    """
    creation of test routes files
    :param source_directory: directory that contains the train routes
    :param target_directory: target path that the test files will be save there
    :return:
    """

    flight_files = get_subdirectories(source_directory)

    for flight in flight_files:

        current_flight_path = os.path.join(source_directory, flight)

        routes_files = get_subdirectories(current_flight_path)

        for index, route in enumerate(routes_files):

            full_route_name = str(flight) + "_Route_" + str(index)
            route_dir = os.path.join(current_flight_path, route)

            create_directories(
                f'{target_directory}/{full_route_name}')  # create the route folder in the target directory

            train_file_df = pd.read_csv(f'{route_dir}/{route}.csv')

            num_of_way_points = len(train_file_df["NameOfWpt"]) - 4

            for j in range(1):
                create_test_set(num_of_way_points=num_of_way_points,
                                directory_path=f'{target_directory}/{full_route_name}',
                                file_name="Mixed_Attack",
                                spoofed_way_point=random.randint(num_of_way_points - 3, num_of_way_points - 1),
                                attack="Mixed",
                                source_df=train_file_df)

                create_test_set(num_of_way_points=num_of_way_points,
                                directory_path=f'{target_directory}/{full_route_name}',
                                file_name="Velocity_Attack",
                                spoofed_way_point=random.randint(num_of_way_points - 3, num_of_way_points - 1),
                                attack="Velocity",
                                source_df=train_file_df)

                create_test_set(num_of_way_points=num_of_way_points,
                                directory_path=f'{target_directory}/{full_route_name}',
                                file_name="Height_Attack",
                                spoofed_way_point=random.randint(num_of_way_points - 3, num_of_way_points - 1),
                                attack="Height",
                                source_df=train_file_df)

                create_test_set(num_of_way_points=num_of_way_points,
                                directory_path=f'{target_directory}/{full_route_name}',
                                file_name="Constant_Attack",
                                spoofed_way_point=random.randint(num_of_way_points - 3, num_of_way_points - 1),
                                attack="Constant",
                                source_df=train_file_df)


def move_test_files_to_target_path(source_directory, target_directory, test_folder_name, attack_name):
    """
    move test files from source path to target path
    :param source_directory: source path
    :param target_directory: target path
    :param test_folder_name: the name of the route
    :param attack_name: the name of the attack
    :return:
    """

    flight_files = get_subdirectories(source_directory)

    for index, route in enumerate(flight_files):
        current_directory = os.path.join(source_directory, route)
        sensors_file = get_sensors_file(current_directory)
        middle_target_directory = os.path.join(target_directory, test_folder_name)
        create_directories(middle_target_directory)
        full_path = os.path.join(middle_target_directory, attack_name)
        create_directories(full_path)
        move_file_to_target_path(current_directory, full_path,
                                 sensors_file, "sensors_{0}.csv".format(index))


# source_path = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\lior\\train_set_routes_data"
# target_path = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\lior\\test_set_routes_data"
# execute_creation(source_path, target_path)

source_folder = "C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\FINAL_SIMULATOR_LATEST+TRAIN+TEST SETS\\test_latest_files"
logs_folder = "C:\\Users\\Lior\\Desktop\\Simulator Versions\\Simulator\\Logs"
target_folder = "C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\FINAL_SIMULATOR_LATEST+TRAIN+TEST SETS\\test_set"

# velocity_arg = ['Up' , 'Down' ,'Stable' ]
# height_arg = ['Up' , 'Down' ,'Stable' ]
# attack_name = ['Constant_Attack' , 'Height_Attack' ,'Mixed_Attack', 'Velocity_Attack']
# index = [0, 1]

velocity_arg = 'None'  # Note: change it just for run the script and replace it back to None!
height_arg = 'None'  # Note: change it just for run the script and replace it back to None!
attack_name = 'None'  # Note: change it just for run the script and replace it back to None!
index = None  # Note: change it just for run the script and replace it back to None!

test_folder_name = '[Velocity = {0}]_[Height = {1}]_Route_{2}'.format(velocity_arg, height_arg, index)

# You should delete logs folder before each run!
# move_test_files_to_target_path(source_directory=logs_folder,
#                                target_directory=target_folder,
#                                test_folder_name=test_folder_name,
#                                attack_name=attack_name)
