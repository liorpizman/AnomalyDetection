import os

import pandas as pd

from gui.shared.helper_methods import load_anomaly_detection_list
from models.random_forest.random_forest_hyper_parameters import random_forest_hyper_parameters
from models.knn.knn_hyper_parameters import knn_hyper_parameters
from models.svr.svr_hyper_parameters import svr_hyper_parameters
from utils.constants import ATTACK_COLUMN
from utils.helper_methods import get_subdirectories
from models.lstm.lstm_hyper_parameters import lstm_hyper_parameters


class InputSettings:
    TRAINING_DATA_PATH = ""
    TEST_DATA_PATH = ""
    RESULTS_DATA_PATH = ""
    ALGORITHMS = set()
    SIMILARITY_SCORES = set()
    SAVE_MODEL = False
    NEW_MODEL_RUNNING = False
    EXISTING_ALGORITHMS = dict()
    FEATURES_COLUMNS_OPTIONS = []
    USERS_SELECTED_FEATURES = dict()
    THREADS = []
    RESULTS_METRICS_DATA = dict()
    FLIGHT_ROUTES = []

    RESULTS_TABLE_ALGORITHM = ""
    RESULTS_TABLE_FLIGHT_ROUTE = ""

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
        set_function = InputSettings.get_algorithm_set_function(algorithm_name)
        set_function(algorithm_parameters)

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
    def init_results_metrics_data():
        InputSettings.RESULTS_METRICS_DATA = dict()

    @staticmethod
    def update_results_metrics_data(updated_dic):
        InputSettings.RESULTS_METRICS_DATA = updated_dic

    @staticmethod
    def get_results_metrics_data():
        return InputSettings.RESULTS_METRICS_DATA

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
        test_columns = list(df_test.columns)
        test_columns.remove(ATTACK_COLUMN)
        InputSettings.FEATURES_COLUMNS_OPTIONS = test_columns

    @staticmethod
    def get_users_selected_features():
        return InputSettings.USERS_SELECTED_FEATURES

    @staticmethod
    def set_users_selected_features(algorithm_name, features_list):
        InputSettings.USERS_SELECTED_FEATURES[algorithm_name] = features_list

    @staticmethod
    def add_new_thread(new_thread):
        InputSettings.THREADS.append(new_thread)

    @staticmethod
    def get_existing_thread():
        current_thread = InputSettings.THREADS[0]
        InputSettings.THREADS = []
        return current_thread

    @staticmethod
    def remove_algorithm(algorithm_name):
        if algorithm_name in InputSettings.ALGORITHMS:
            InputSettings.ALGORITHMS.remove(algorithm_name)

    @staticmethod
    def set_Random_Forest(algorithm_parameters):
        InputSettings.ALGORITHMS.add("Random Forest")
        for param in algorithm_parameters:
            Random_Forest_setting_function = getattr(random_forest_hyper_parameters, "set_" + param)
            Random_Forest_setting_function(algorithm_parameters[param])

    @staticmethod
    def set_SVR(algorithm_parameters):
        InputSettings.ALGORITHMS.add("SVR")
        for param in algorithm_parameters:
            svr_setting_function = getattr(svr_hyper_parameters, "set_" + param)
            svr_setting_function(algorithm_parameters[param])

    @staticmethod
    def set_KNN(algorithm_parameters):
        InputSettings.ALGORITHMS.add("KNN")
        for param in algorithm_parameters:
            knn_setting_function = getattr(knn_hyper_parameters, "set_" + param)
            knn_setting_function(algorithm_parameters[param])

    @staticmethod
    def get_algorithm_set_function(algorithm_name):
        algorithms = load_anomaly_detection_list()
        switcher = {
            algorithms[0]: InputSettings.set_LSTM,
            algorithms[1]: InputSettings.set_SVR,
            algorithms[2]: InputSettings.set_KNN,
            algorithms[3]: InputSettings.set_Random_Forest
        }
        return switcher.get(algorithm_name, None)

    @staticmethod
    def set_flight_routes(flight_routes):
        InputSettings.FLIGHT_ROUTES = flight_routes

    @staticmethod
    def get_flight_routes():
        return InputSettings.FLIGHT_ROUTES

    @staticmethod
    def set_results_selected_algorithm(selected_algorithm):
        InputSettings.RESULTS_TABLE_ALGORITHM = selected_algorithm

    @staticmethod
    def set_results_selected_flight_route(selected_flight_route):
        InputSettings.RESULTS_TABLE_FLIGHT_ROUTE = selected_flight_route

    @staticmethod
    def get_results_selected_algorithm():
        return InputSettings.RESULTS_TABLE_ALGORITHM

    @staticmethod
    def get_results_selected_flight_route():
        return InputSettings.RESULTS_TABLE_FLIGHT_ROUTE
