import os

import win32api


def new_model_train_path_validation(train_path):
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
        elif level > 1:  # level > 1
            return False
    return routs


def test_path_validation(test_path, input_routes):
    for root, dirs, files in os.walk(test_path):
        level = root.replace(test_path, '').count(os.sep)
        if level == 0 and ((input_routes and dirs != input_routes) or files):  # first level in path
            return False
        elif level == 1 and (not dirs or files):  # second level in path
            return False
        elif level == 2:  # third level in path
            if dirs or not files:  # if there are subdirectories , or there is no attack route file in the second level
                return False
            for f in files:
                if not f.endswith('.csv') or not f[:-4].isdigit():
                    return False  # if there are files other than "csv" type , or their name are not represents numbers
        elif level > 2:  # level > 2
            return False
    return True


def new_model_paths_structure_validation(train_path, test_path):
    routes = new_model_train_path_validation(train_path)
    if routes:
        return test_path_validation(test_path, routes)
    else:
        return False


def load_model_paths_structure_validation(test_path):
    return test_path_validation(test_path, None)


def is_unique_paths(paths):
    """
    check if paths are uniques
    :param paths: list of paths
    :return:
    """
    if len(set(paths)) == len(paths):
        return True
    return False


def new_model_paths_validation(train_path, test_path, results_path):
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
    return os.path.exists(os.path.dirname(path))


def is_valid_directories(paths):
    """
    check if all the paths are valids
    :param paths:list of paths
    :return:
    """
    for path in paths:
        if not os.path.exists(os.path.dirname(path)):
            return False
    return True


def is_valid_model_paths(paths):
    """
    validats that each path is an h5 valid model file
    :param paths:list of paths
    :return:
    """
    for path in paths:
        if not os.path.exists(os.path.dirname(path)):
            return False
        files = os.listdir(path)
        for file in files:
            fullPath = os.path.join(path, file)
            if not os.path.isfile(fullPath) or not (file.endswith('.h5') or (file == "model_data.json")):
                return False
        return True
