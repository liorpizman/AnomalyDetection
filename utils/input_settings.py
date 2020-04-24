'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Main values storage in order to get updated values in each action at the system
'''

import os
import pandas as pd

from gui.shared.helper_methods import load_anomaly_detection_list
from models.mlp.mlp_hyper_parameters import mlp_hyper_parameters
from models.random_forest.random_forest_hyper_parameters import random_forest_hyper_parameters
from models.svr.svr_hyper_parameters import svr_hyper_parameters
from utils.constants import COLUMNS_TO_REMOVE
from utils.helper_methods import get_subdirectories
from models.lstm.lstm_hyper_parameters import lstm_hyper_parameters


class InputSettings:
    """
    A Class used to manage all the global values in the system
    """

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
    RESULTS_TABLE_SIMILARITY_FUNCTION = ""

    """
    Attributes
    ----------
    
    TRAINING_DATA_PATH                   : str

    TEST_DATA_PATH                       : str
            
    RESULTS_DATA_PATH                    : str
            
    ALGORITHMS                           : set
            
    SIMILARITY_SCORES                    : set
            
    SAVE_MODEL                           : bool
            
    NEW_MODEL_RUNNING                    : bool
            
    EXISTING_ALGORITHMS                  : dict
            
    FEATURES_COLUMNS_OPTIONS             : list
            
    USERS_SELECTED_FEATURES              : dict
            
    THREADS                              : list
            
    RESULTS_METRICS_DATA                 : dict
            
    FLIGHT_ROUTES                        : list
            
    RESULTS_TABLE_ALGORITHM              : str
            
    RESULTS_TABLE_FLIGHT_ROUTE           : str
    
    RESULTS_TABLE_SIMILARITY_FUNCTION    : str

    Methods
    -------
    set_training_data_path(path)
            Description | Set the path of the train data set directory
            
    get_training_data_path()
            Description | Get the path of the train data set directory
            
    set_test_data_path(path)
            Description | Set the path of the test data set directory
            
    get_test_data_path()
            Description | Get the path of the test data set directory
            
    set_results_path(path)
            Description | Set the path of the results directory
            
    get_results_path()
            Description | Get the path of the results directory
            
    get_algorithms()
            Description | Get all the algorithms
            
    get_similarity()
            Description | Get all similarity scores
            
    set_algorithm_parameters(algorithm_name, algorithm_parameters)
            Description | Set the parameters which were chosen by the user to a given algorithm
            
    set_LSTM(algorithm_parameters)
            Description | Set the parameters which were chosen by the user for LSTM
            
    remove_algorithm_parameters(algorithm_name, algorithm_parameters)
            Description | Remove the parameters which were chosen by the user to a given algorithm
            
    remove_LSTM(algorithm_parameters)
            Description | Remove the parameters which were chosen by the user for LSTM
            
    set_similarity_score(similarity_list)
            Description | Set the list of all chosen similarity function by the user
            
    set_saving_model(save_model)
            Description | Set the variable which indicates whether the user want to save the current model or not
            
    get_saving_model()
            Description | Get the variable which indicates whether the user want to save the current model or not
            
    set_new_model_running(new_model_running)
             Description | Set the variable which indicates whether the user chose to create new model or to load an
                          existing model
            
    get_new_model_running()
            Description | Get the variable which indicates whether the user chose to create new model or to load an
                          existing model
            
    set_existing_algorithms(existing_algorithms)
            Description | Set a dictionary which includes all the algorithm which were chosen by the user in a load
                          existing models flow
            
    get_existing_algorithms()
            Description | Get a dictionary which includes all the algorithm which were chosen by the user in a load
                          existing models flow
            
    get_existing_algorithm_path(algorithm_name)
            Description | Get the path for an existing algorithm in a load existing models flow
            
    init_results_metrics_data()
            Description | Init the values for all the metrics which were set in the previous process
            
    update_results_metrics_data(updated_dic)
            Description | Update the values of all the metrics for the current flow
            
    get_results_metrics_data()
            Description |  Get the dictionary which includes all the metrics for the current flow
            
    get_features_columns_options()
            Description | Get the data set columns which were loaded from the test data set
            
    set_features_columns_options()
            Description | Set the data set columns which were loaded from the test data set
            
    get_users_selected_features()
            Description | Get the data set columns which were selected by the user
            
    set_users_selected_features(features_list)
            Description | Set the data set columns which were selected by the user
            
    add_new_thread(new_thread)
            Description | Add new running thread to the system
            
    get_existing_thread()
            Description | Get running thread
            
    remove_algorithm(algorithm_name)
            Description | Remove a given algorithm
            
    set_Random_Forest(algorithm_parameters)
            Description | Set the parameters which were chosen by the user for Random forest
            
    set_SVR(algorithm_parameters)
            Description | Set the parameters which were chosen by the user for SVR
            
    set_MLP(algorithm_parameters)
            Description | Set the parameters which were chosen by the user for MLP
            
    get_algorithm_set_function(algorithm_name)
            Description | Switch to get the set function for a given algorithm
            
    set_flight_routes(flight_routes)
            Description | Set all the flight routes which are in the test data set
            
    get_flight_routes()
            Description | Get all the flight routes which are in the test data set
            
    set_results_selected_algorithm(selected_algorithm)
            Description | Set the variable which indicates which algorithm should be shown in the results table
                          at this moment
            
    set_results_selected_flight_route(selected_flight_route)
            Description | Set the variable which indicates which flight route should be shown in the results table
                          at this moment
            
    get_results_selected_algorithm()
            Description | Get the variable which indicates which algorithm should be shown in the results table
                          at this moment
            
    get_results_selected_flight_route()
            Description | Get the variable which indicates which flight route should be shown in the results table
                          at this moment
            
    reset_input_settings_params()
            Description | Reset all the values of input settings attributes
            
    set_results_selected_similarity_function(similarity_function)
            Description | Set the variable which indicates which similarity function should be shown in the results
                          table at this moment
                          
    get_results_selected_similarity_function()
            Description | Get the variable which indicates which similarity function should be shown in the results 
                          table at this moment

    """

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

        # Iterate over all parameters for LSTM algorithm
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

        # Iterate over all parameters for LSTM algorithm
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

        # Get the columns in test data set in order to do feature selection by the user
        test_data_path = InputSettings.get_test_data_path()
        flight_route = get_subdirectories(test_data_path).__getitem__(0)
        flight_dir = os.path.join(test_data_path, flight_route)
        attack = get_subdirectories(flight_dir).__getitem__(0)
        flight_csv = os.listdir(f'{test_data_path}/{flight_route}/{attack}').__getitem__(0)
        df_test = pd.read_csv(f'{test_data_path}/{flight_route}/{attack}/{flight_csv}')
        test_columns = list(df_test.columns)

        # Cleaning meta-data - Remove columns from a yaml file, such as: index, flight_id etc.
        for column in COLUMNS_TO_REMOVE:
            if column in test_columns:
                test_columns.remove(column)

        InputSettings.FEATURES_COLUMNS_OPTIONS = test_columns

    @staticmethod
    def get_users_selected_features():
        return InputSettings.USERS_SELECTED_FEATURES

    @staticmethod
    def set_users_selected_features(features_list):
        InputSettings.USERS_SELECTED_FEATURES = dict()
        for algorithm in InputSettings.ALGORITHMS:
            InputSettings.USERS_SELECTED_FEATURES[algorithm] = features_list

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

        # Iterate over all parameters for Random Forest algorithm
        for param in algorithm_parameters:
            Random_Forest_setting_function = getattr(random_forest_hyper_parameters, "set_" + param)
            Random_Forest_setting_function(algorithm_parameters[param])

    @staticmethod
    def set_SVR(algorithm_parameters):
        InputSettings.ALGORITHMS.add("SVR")

        # Iterate over all parameters for SVR algorithm
        for param in algorithm_parameters:
            svr_setting_function = getattr(svr_hyper_parameters, "set_" + param)
            svr_setting_function(algorithm_parameters[param])

    @staticmethod
    def set_MLP(algorithm_parameters):
        InputSettings.ALGORITHMS.add("MLP")

        # Iterate over all parameters for MLP algorithm
        for param in algorithm_parameters:
            mlp_setting_function = getattr(mlp_hyper_parameters, "set_" + param)
            mlp_setting_function(algorithm_parameters[param])

    @staticmethod
    def get_algorithm_set_function(algorithm_name):
        algorithms = load_anomaly_detection_list()

        # Switch to get the suitable set function for a given algorithm
        switcher = {
            algorithms[0]: InputSettings.set_LSTM,
            algorithms[1]: InputSettings.set_SVR,
            algorithms[2]: InputSettings.set_MLP,
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

    @staticmethod
    def reset_input_settings_params():

        # Reset all global attributes
        InputSettings.TRAINING_DATA_PATH = ""
        InputSettings.TEST_DATA_PATH = ""
        InputSettings.RESULTS_DATA_PATH = ""
        InputSettings.ALGORITHMS = set()
        InputSettings.SIMILARITY_SCORES = set()
        InputSettings.SAVE_MODEL = False
        InputSettings.NEW_MODEL_RUNNING = False
        InputSettings.EXISTING_ALGORITHMS = dict()
        InputSettings.FEATURES_COLUMNS_OPTIONS = []
        InputSettings.USERS_SELECTED_FEATURES = dict()
        InputSettings.THREADS = []
        InputSettings.RESULTS_METRICS_DATA = dict()
        InputSettings.FLIGHT_ROUTES = []

        InputSettings.RESULTS_TABLE_ALGORITHM = ""
        InputSettings.RESULTS_TABLE_FLIGHT_ROUTE = ""

    @staticmethod
    def set_results_selected_similarity_function(similarity_function):
        InputSettings.RESULTS_TABLE_SIMILARITY_FUNCTION = similarity_function

    @staticmethod
    def get_results_selected_similarity_function():
        return InputSettings.RESULTS_TABLE_SIMILARITY_FUNCTION
