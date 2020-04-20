#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Script to create test set data for different routes by the simulator

for create test set you have to :
1. run -> execute_creation function
2. run -> create_attack_files function
3. you have to select specific route and attack - > run the suitable files in the simulator
4. run -> move_test_files_to_target_path function
5. delete files from the simulator logs folder
6. back to stage 3
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


def longitude_column_creation(num_of_way_points):
    """
    create Longitude column
    :param num_of_way_points: num of way points in route
    :return:Longitude column's data
    """

    return [random.uniform(34.660001, 34.959999) if index <= num_of_way_points or index == num_of_way_points + 3
            else "-"
            for index in range(num_of_way_points + 6)]


def latitude_column_creation(num_of_way_points):
    """
    create Latitude column
    :param num_of_way_points: num of way points in route
    :return:Latitude column's data
    """

    return [random.uniform(31.888881, 32.009001) if index <= num_of_way_points or index == num_of_way_points + 3
            else "-"
            for index in range(num_of_way_points + 6)]


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

    for index in range(num_of_way_points + 6):

        if index == 0:
            column.append("0")
        elif 1 <= index <= num_of_way_points or index == num_of_way_points + 3:
            base = random.randint(1, 3)
            column.append(last_height)
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
            for index in range(num_of_way_points + 6)]


def velocity_column_creation(num_of_way_points, state):
    """
    create Velocity column
    :param num_of_way_points: num of way points in route
    :param state: can be 'Up' , 'Down' ro 'Stable'
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

    for index in range(num_of_way_points + 6):

        if 1 <= index <= num_of_way_points or index == num_of_way_points + 3:
            base = random.randint(1, 2)
            column.append(last_velocity)
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
        height_source_column[spoofer_index] = height_source_column[way_point_index]
        velocity_source_column[spoofer_index] = velocity_source_column[way_point_index]

    elif attack == 'Height':
        height_source_column[spoofer_index] = height_source_column[way_point_index] + height_factor * height_base
        velocity_source_column[spoofer_index] = velocity_source_column[way_point_index]

    elif attack == 'Velocity':
        height_source_column[spoofer_index] = height_source_column[way_point_index]
        velocity_source_column[spoofer_index] = velocity_source_column[
                                                    way_point_index] + velocity_factor * velocity_base

    else:  # attack == 'Mixed'
        height_source_column[spoofer_index] = height_source_column[way_point_index] + height_factor * height_base
        velocity_source_column[spoofer_index] = velocity_source_column[
                                                    way_point_index] + velocity_factor * velocity_base

    return height_source_column, velocity_source_column


def generate_height_and_velocity_columns(num_of_way_points,
                                         height_state,
                                         velocity_state,
                                         attack,
                                         spoofed_way_point,
                                         ):
    """
    create Height and Velocity columns
    :param num_of_way_points: num of way points in route
    :param height_state: can be 'Up' , 'Down' or 'Stable':
    :param velocity_state: can be 'Up' , 'Down' or 'Stable':
    :param attack: can be 'Constant' , 'Height' , 'Velocity' or 'Mixed'
    :param spoofed_way_point: start point of the spoofing attack
    :return: Height and velocity columns
    """

    height_source_column = height_column_creation(num_of_way_points, height_state)
    velocity_source_column = velocity_column_creation(num_of_way_points, velocity_state)
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


def move_test_files_to_target_path(source_directory, target_directory):
    """
    move test files from source path to target path
    :param source_directory: source path
    :param target_directory: target path
    :return:
    """

    flight_files = get_subdirectories(source_directory)

    for index, rout in enumerate(flight_files):
        current_directory = os.path.join(source_directory, rout)
        sensors_file = get_sensors_file(current_directory)
        move_file_to_target_path(current_directory, target_directory,
                                 sensors_file, "sensors_{0}.csv".format(index))


def create_attack_files(source_directory, files_amount):
    """
    create attack files folders
    :param source_directory: source path
    :param files_amount: files amount
    :return:
    """

    for index in range(files_amount):
        rout_name = "rout_" + str(index)
        create_directories(f'{source_directory}/{rout_name}')
        for attack in ['Constant Attack', 'Height Attack', 'Velocity Attack', 'Mixed Attack']:
            create_directories(f'{source_directory}/{rout_name}/{attack}')


def create_test_set(num_of_way_points,
                    velocity_state,
                    height_state,
                    directory_path,
                    file_name,
                    spoofed_way_point,
                    attack):
    """
    creation of basic route
    :param num_of_way_points: num of way points in route
    :param velocity_state: can be 'Up' , 'Down' ro 'Stable'
    :param height_state: can be 'Up' , 'Down' ro 'Stable'
    :param directory_path: file location
    :param file_name: route name
    :param spoofed_way_point: start point of the spoofing attack
    :param attack: can be 'Constant' , 'Height' , 'Velocity' or 'Mixed'
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
    df["Frequency"] = frequency_column_creation(num_of_way_points)
    df["ParabolicAntennaTypeIndex"] = parabolic_antenna_column_creation(num_of_way_points)
    df["OmniAntennaTypeIndex"] = omni_antenna_column_creation(num_of_way_points)
    df["IpOfDrone"] = ip_of_drone_column_creation(num_of_way_points)
    df["IsNoLinkActiveInWpt"] = no_link_column_creation(num_of_way_points)
    df["IsSpoofingActivatedInWpt"] = spoofing_activated_column_creation(num_of_way_points, spoofed_way_point)
    df["ForwardType"] = forward_type_column_creation(num_of_way_points)
    df["NameOfSpoofer"] = name_of_spoofer_column_creation(num_of_way_points)

    df["Height"], df["Velocity"] = generate_height_and_velocity_columns(num_of_way_points,
                                                                        height_state,
                                                                        velocity_state,
                                                                        attack,
                                                                        spoofed_way_point,
                                                                        )

    write_data_frame_to_csv(directory_path=directory_path,
                            file_name=file_name,
                            data_frame=df)


def execute_creation(source_folder, files_amount):
    """
    creation of test routes files
    :param source_folder: source path
    :param files_amount: files amount
    :return:
    """

    for i in range(files_amount):
        rout_name = "rout_" + str(i)
        create_directories(f'{source_folder}/{rout_name}')
    for i in range(files_amount):
        rout_name = "rout_" + str(i)
        num_of_way_points = random.randint(10, 20)
        for j in range(2):
            create_test_set(num_of_way_points=num_of_way_points,
                            velocity_state=get_random_state(),
                            height_state=get_random_state(),
                            directory_path=f'{source_folder}/{rout_name}',
                            file_name="Mixed_Attack_{0}".format(j),
                            spoofed_way_point=random.randint(6, num_of_way_points - 2),
                            attack="Mixed")

            create_test_set(num_of_way_points=num_of_way_points,
                            velocity_state=get_random_state(),
                            height_state=get_random_state(),
                            directory_path=f'{source_folder}/{rout_name}',
                            file_name="Velocity_Attack_{0}".format(j),
                            spoofed_way_point=random.randint(6, num_of_way_points - 2),
                            attack="Velocity")

            create_test_set(num_of_way_points=num_of_way_points,
                            velocity_state=get_random_state(),
                            height_state=get_random_state(),
                            directory_path=f'{source_folder}/{rout_name}',
                            file_name="Constant_Attack_{0}".format(j),
                            spoofed_way_point=random.randint(6, num_of_way_points - 2),
                            attack="Constant")

            create_test_set(num_of_way_points=num_of_way_points,
                            velocity_state=get_random_state(),
                            height_state=get_random_state(),
                            directory_path=f'{source_folder}/{rout_name}',
                            file_name="Height_Attack_{0}".format(j),
                            spoofed_way_point=random.randint(6, num_of_way_points - 2),
                            attack="Height")


# source_folder = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\our_creation\\routs_data\\test_data"
# logs_folder = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\Simulator\\Logs"
# target_folder = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\our_creation\\test"
# files_amount = 6
#
# execute_creation(source_folder=source_folder, files_amount=files_amount)
# create_attack_files(source_directory=target_folder, files_amount=files_amount)
# attack_folder = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\our_creation\\test\\rout_6\\Constant Attack"
# move_test_files_to_target_path(source_directory=logs_folder, target_directory=attack_folder)
