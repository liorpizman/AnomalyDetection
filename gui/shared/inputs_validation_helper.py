'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Methods to handle repeatable validations which are done by the gui controller
'''

import os
import win32api

from gui.shared.helper_methods import read_json_file


def new_model_train_path_validation(train_path):
    """
    Validation of train data set directory
    :param train_path: train path directory
    :return: True if the input valid , otherwise false
    """

    routs = list()
    without_anomaly_file_name = "without_anom.csv"

    for root, dirs, files in os.walk(train_path):
        level = root.replace(train_path, '').count(os.sep)

        if level == 0:  # first level in path
            if files or not dirs:  # if there is files , or there is no subdirectories in the first level
                return False
            else:
                routs = dirs
        elif level == 1 and (dirs or without_anomaly_file_name not in files):  # second level in path
            return False  # if there are subdirectories , or there is no "without_anom" file in the second level
        elif level > 1:
            return False

    return routs


def test_path_validation(test_path, input_routes):
    """
    Validation of test data set directory
    :param test_path: test path directory
    :param input_routes: path of input routes
    :return: True if the input valid , otherwise false
    """

    for root, dirs, files in os.walk(test_path):
        level = root.replace(test_path, '').count(os.sep)

        if level == 0 and ((input_routes and dirs != input_routes) or files):  # first level in path
            return False
        elif level == 1 and (not dirs or files):  # second level in path
            return False
        elif level == 2:  # third level in path
            if dirs or not files:  # if there are subdirectories , or there is no attack route file in the second level
                return False
            try:
                for f in files:
                    splitted_file = f.split("_")
                    sensors = splitted_file.__getitem__(0)
                    flight_index = splitted_file.__getitem__(1)
                    # if there are files other than "csv" type , or their name are not in "sensors_[index]" format
                    if sensors != "sensors" \
                            or not flight_index \
                            or not flight_index.endswith('.csv') \
                            or not flight_index[:-4].isdigit():
                        return False
            except:
                return False
        elif level > 2:
            return False

    return True


def new_model_paths_structure_validation(train_path, test_path):
    """
    Validation of new model flow inputs structure
    :param train_path: the path of train data set directory
    :param test_path: the path of test data set directory
    :return: True if the input valid , otherwise false
    """

    routes = new_model_train_path_validation(train_path)

    if routes:
        return test_path_validation(test_path, routes)
    else:
        return False


def load_model_paths_structure_validation(test_path):
    """
    Validation of load existing model flow inputs structure
    :param test_path: the path of test data set directory
    :return: True if the input valid , otherwise false
    """

    return test_path_validation(test_path, None)


def is_unique_paths(paths):
    """
    check if paths are unique
    :param paths: list of paths
    :return: True if the input valid , otherwise false
    """

    if len(set(paths)) == len(paths):
        return True

    return False


def new_model_paths_validation(train_path, test_path, results_path):
    """
    Validation of all paths in the new model creation flow
    :param train_path: train data set directory path
    :param test_path: test data set directory path
    :param results_path: results directory path
    :return: True if the input valid , otherwise false
    """

    if not is_valid_directories([train_path, test_path, results_path]):
        win32api.MessageBox(0, 'At least one of your inputs is invalid!', 'Invalid inputs', 0x00001000)
        return False
    elif not is_unique_paths([train_path, test_path, results_path]):
        win32api.MessageBox(0, 'At least two of your inputs are identical!.', 'Identical inputs', 0x00001000)
        return False
    elif not new_model_paths_structure_validation(train_path, test_path):
        win32api.MessageBox(0, 'At least one of your inputs is '
                               'not in the appropriate file structure. Please go over the "README" file for more details.',
                            'Invalid inputs structure', 0x00001000)
        return False

    return True


def load_model_paths_validation(test_path, results_path):
    """
    Validation of all paths in the load existing model creation flow
    :param test_path: test data set directory path
    :param results_path: results directory path
    :return: True if the input valid , otherwise false
    """

    if not is_valid_directories([test_path, results_path]):
        win32api.MessageBox(0, 'At least one of your inputs is invalid!', 'Invalid inputs', 0x00001000)
        return False
    elif not is_unique_paths([test_path, results_path]):
        win32api.MessageBox(0, 'At least two of your inputs are identical!.', 'Identical inputs', 0x00001000)
        return False
    elif not load_model_paths_structure_validation(test_path):
        win32api.MessageBox(0, 'At least one of your inputs is '
                               'not in the appropriate file structure. Please go over the "README" file for more details.',
                            'Invalid inputs structure', 0x00001000)
        return False

    return True


def is_valid_directory(path):
    """
    Checks whether a path contains a valid directory or not
    :param path: path of a directory
    :return: True if the input valid , otherwise false
    """

    return os.path.exists(os.path.dirname(path))


def is_valid_directories(paths):
    """
    Checks whether all paths contain valid directories or not
    :param paths: list of paths
    :return: True if the input valid , otherwise false
    """

    for path in paths:
        if not os.path.exists(os.path.dirname(path)):
            return False

    return True


def is_valid_model_paths(paths):
    """
    Validation for each path in order to ensure that it contains a valid model file
    :param paths: list of paths
    :return: True if the input valid , otherwise false
    """

    for path in paths:
        if not os.path.exists(os.path.dirname(path)):
            return False
        files = os.listdir(path)
        models_counter = 0
        scalars_counter = 0
        json_files_counter = 0
        for file in files:
            fullPath = os.path.join(path, file)
            if os.path.isfile(fullPath):
                if file.endswith('.h5') or file.endswith('_model.pkl'):
                    models_counter += 1
                if file.endswith('_scaler.pkl'):
                    scalars_counter += 1
                if (file == "model_data.json"):
                    json_files_counter += 1
        if not (models_counter == 1 and scalars_counter == 2 and json_files_counter == 1):
            return False
    return True


def is_valid_model_data_file(paths):
    """
    Validation for each path in order to ensure that it contains JSON file with features and threshold data
    :param paths: list of paths
    :return: True if the input valid , otherwise false
    """

    for path in paths:
        if not os.path.exists(os.path.dirname(path)):
            return False
        files = os.listdir(path)
        for file in files:
            if file != "model_data.json":
                continue
            full_file_Path = os.path.join(path, file)
            if not os.path.isfile(full_file_Path):
                return False
            required_fields = get_json_required_fields()
            algorithm_json_file = read_json_file(full_file_Path)
            for field in required_fields:
                if field not in algorithm_json_file or not algorithm_json_file[field]:
                    return False

    return True


def get_json_required_fields():
    """
    Get required fields in JSON file
    :return: required fields
    """

    return ['features', 'threshold']


def pre_tune_model_path_validation(input_path, results_path):
    """
    Validation for tune model paths in order to handle next step
    :param input_path: file path
    :param results_path: results directory path
    :return: return true if valid, otherwise false
    """

    return (os.path.isfile(input_path) and input_path.endswith('without_anom.csv')) \
           and (os.path.exists(results_path))
