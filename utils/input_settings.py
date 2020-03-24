import os

import pandas as pd

from models.lstm.lstm_hyper_parameters import lstm_hyper_parameters
from utils.helper_methods import get_subdirectories

class InputSettings:
    TRAINING_DATA_PATH = ""
    TEST_DATA_PATH = ""
    RESULTS_DATA_PATH = ""
    ALGORITHMS = set()
    SIMILARITY_SCORES = set()
    SAVE_MODEL = False
    NEW_MODEL_RUNNING = False
    LOAD_MODEL_THRESHOLD = None
    EXISTING_ALGORITHMS = dict()
    FEATURES_COLUMNS_OPTIONS = []
    USERS_SELECTED_FEATURES = []
    THREADS = []

    @staticmethod
    def set_training_data_path(path):
        InputSettings.TRAINING_DATA_PATH = path

    @staticmethod
    def get_training_data_path():
        return InputSettings.TRAINING_DATA_PATH

    @staticmethod
    def set_test_data_path(path):
        InputSettings.TEST_DATA_PATH = path

    @staticmethod
    def get_test_data_path():
        return InputSettings.TEST_DATA_PATH

    @staticmethod
    def set_results_path(path):
        InputSettings.RESULTS_DATA_PATH = path

    @staticmethod
    def get_results_path():
        return InputSettings.RESULTS_DATA_PATH

    @staticmethod
    def get_algorithms():
        return InputSettings.ALGORITHMS

    @staticmethod
    def get_similarity():
        return InputSettings.SIMILARITY_SCORES

    @staticmethod
    def set_algorithm_parameters(algorithm_name, algorithm_parameters):
        algorithm_setting_function = getattr(InputSettings, "set_" + algorithm_name)
        algorithm_setting_function(algorithm_parameters)

    @staticmethod
    def set_LSTM(algorithm_parameters):
        InputSettings.ALGORITHMS.add("LSTM")
        for param in algorithm_parameters:
            lstm_setting_function = getattr(lstm_hyper_parameters, "set_" + param)
            lstm_setting_function(algorithm_parameters[param])

    @staticmethod
    def remove_algorithm_parameters(algorithm_name, algorithm_parameters):
        algorithm_remove_function = getattr(InputSettings, "remove_" + algorithm_name)
        algorithm_remove_function(algorithm_parameters)

    @staticmethod
    def remove_LSTM(algorithm_parameters):
        if "LSTM" not in InputSettings.ALGORITHMS:
            return
        InputSettings.ALGORITHMS.remove("LSTM")
        for param in algorithm_parameters:
            lstm_setting_function = getattr(lstm_hyper_parameters, "remove_" + param)
            lstm_setting_function(algorithm_parameters[param])

    @staticmethod
    def set_similarity_score(similarity_list):
        InputSettings.SIMILARITY_SCORES = similarity_list

    @staticmethod
    def set_saving_model(save_model):
        InputSettings.SAVE_MODEL = save_model

    @staticmethod
    def get_saving_model():
        return InputSettings.SAVE_MODEL

    @staticmethod
    def set_new_model_running(new_model_running):
        InputSettings.NEW_MODEL_RUNNING = new_model_running

    @staticmethod
    def get_new_model_running():
        return InputSettings.NEW_MODEL_RUNNING

    @staticmethod
    def set_existing_algorithms(existing_algorithms):
        InputSettings.EXISTING_ALGORITHMS = existing_algorithms

    @staticmethod
    def get_existing_algorithms():
        return InputSettings.EXISTING_ALGORITHMS

    @staticmethod
    def get_existing_algorithm_path(algorithm_name):
        return InputSettings.EXISTING_ALGORITHMS[algorithm_name]

    @staticmethod
    def get_existing_algorithms_threshold():
        return InputSettings.LOAD_MODEL_THRESHOLD

    @staticmethod
    def set_existing_algorithms_threshold(threshold):
        InputSettings.LOAD_MODEL_THRESHOLD = threshold

    @staticmethod
    def get_features_columns_options():
        return InputSettings.FEATURES_COLUMNS_OPTIONS

    @staticmethod
    def set_features_columns_options():
        test_data_path = InputSettings.get_test_data_path()
        flight_route = get_subdirectories(test_data_path).__getitem__(0)
        flight_dir = os.path.join(test_data_path, flight_route)
        attack = get_subdirectories(flight_dir).__getitem__(0)
        flight_csv = os.listdir(f'{test_data_path}/{flight_route}/{attack}').__getitem__(0)
        df_test = pd.read_csv(f'{test_data_path}/{flight_route}/{attack}/{flight_csv}')
        InputSettings.FEATURES_COLUMNS_OPTIONS = list(df_test.columns)

    @staticmethod
    def get_users_selected_features():
        return InputSettings.USERS_SELECTED_FEATURES

    @staticmethod
    def set_users_selected_features(features_list):
        InputSettings.USERS_SELECTED_FEATURES = features_list

    @staticmethod
    def add_new_thread(new_thread):
        InputSettings.THREADS.append(new_thread)

    @staticmethod
    def get_existing_thread():
        current_thread = InputSettings.THREADS[0]
        InputSettings.THREADS = []
        return current_thread
