'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Switcher between models execution in the system
'''

from gui.shared.helper_methods import read_json_file, get_model_path, load_anomaly_detection_list
from models.lstm.lstm_execution import run_model as run_lstm_model
from models.svr.svr_execution import run_model as run_svr_model
from models.random_forest.random_forest_execution import run_model as run_random_forest_model
from models.linear_regression.linear_regression_execution import run_model as run_linear_regression_model
from utils.input_settings import InputSettings
from utils.helper_methods import get_subdirectories


class ModelsExecution:
    """
    A Class used to execute different machine learning algorithms dynamically

    Methods
    -------
    get_new_model_parameters()
            Description | Get the parameters which belongs to new model flow

    get_load_model_parameters()
            Description | Get the parameters which belongs to load existing model flow

    get_parameters()
            Description | Get parameters which belong to both flows

    run_models()
            Description | Executes all the algorithms which were chosen - suitable for both flows

    LSTM_execution(test_data_path, results_path, similarity_score, training_data_path, save_model, new_model_running,
                   algorithm_path, threshold, features_list)
            Description | Executes Long short-term memory algorithm

    SVR_execution(test_data_path, results_path, similarity_score, training_data_path, save_model, new_model_running,
                  algorithm_path, threshold, features_list)
            Description | Executes Support Vector Regression algorithm

    Random_Forest_execution(test_data_path, results_path, similarity_score, training_data_path, save_model,
                            new_model_running, algorithm_path, threshold, features_list)
            Description | Executes Random forest algorithm

    get_algorithm_execution_function(algorithm_name)
            Description | Switch to get the execution function for a given algorithm

    """

    @classmethod
    def get_new_model_parameters(cls):
        """
        get the parameters which belong to new model flow
        :return: new model parameters
        """

        return (InputSettings.get_training_data_path(),
                InputSettings.get_saving_model(),
                InputSettings.get_algorithms(),
                None,
                InputSettings.get_users_selected_features(),)

    @classmethod
    def get_load_model_parameters(cls):
        """
        get the parameters which belong to load existing model flow
        :return: existing model parameters
        """

        return (None,
                False,
                InputSettings.get_existing_algorithms(),
                None)

    @classmethod
    def get_parameters(cls):
        """
        get parameters which belong to both flows
        :return: both flows parameters
        """

        return (InputSettings.get_similarity(),
                InputSettings.get_test_data_path(),
                InputSettings.get_results_path(),
                InputSettings.get_new_model_running(),)

    @staticmethod
    def run_models():
        """
        executes all the algorithms which were chosen - suitable for both flows
        :return:
        """

        similarity_score, test_data_path, results_path, new_model_running = ModelsExecution.get_parameters()

        if new_model_running:
            training_data_path, save_model, algorithms, threshold, features_list = ModelsExecution.get_new_model_parameters()
        else:
            training_data_path, save_model, algorithms, threshold = ModelsExecution.get_load_model_parameters()

        # Init evaluation metrics data which will be presented in the results table
        InputSettings.init_results_metrics_data()

        # Set test data - flight routes
        flight_routes = get_subdirectories(test_data_path)
        InputSettings.set_flight_routes(flight_routes)

        for algorithm in algorithms:

            # Set new nested dictionary for a chosen algorithm
            results_data = InputSettings.get_results_metrics_data()
            results_data[algorithm] = dict()
            InputSettings.update_results_metrics_data(results_data)

            # Checks whether the current flow in the system is new model creation or loading an existing model
            if new_model_running:
                algorithm_model_path = None
                algorithm_features_list = features_list[algorithm]
            else:
                algorithm_path = InputSettings.get_existing_algorithm_path(algorithm)
                algorithm_json_file = read_json_file(f'{algorithm_path}/model_data.json')
                algorithm_features_list = algorithm_json_file['features']
                threshold = algorithm_json_file['threshold']
                algorithm_model_path = get_model_path(algorithm_path)

            # Dynamic execution for each chosen model
            model_execution_function = ModelsExecution.get_algorithm_execution_function(algorithm)
            model_execution_function(test_data_path,
                                     results_path,
                                     similarity_score,
                                     training_data_path,
                                     save_model,
                                     new_model_running,
                                     algorithm_model_path,
                                     threshold,
                                     algorithm_features_list)

    @staticmethod
    def LSTM_execution(test_data_path,
                       results_path,
                       similarity_score,
                       training_data_path,
                       save_model,
                       new_model_running,
                       algorithm_path,
                       threshold,
                       features_list):
        """
        executes Long short-term memory algorithm
        :param test_data_path: path of test data set directory
        :param results_path: path of results  directory
        :param similarity_score: similarity functions
        :param training_data_path: path of train data set directory
        :param save_model: Indicator for saving the model or not
        :param new_model_running: Indicator whether the current flow is new model creation or not
        :param algorithm_path: path of existing model directory
        :param threshold: which was calculated in an existing model
        :param features_list: all the features in the test data set
        :return: results after model prediction
        """

        # Run LSTM function
        run_lstm_model(training_data_path,
                       test_data_path,
                       results_path,
                       similarity_score,
                       save_model,
                       new_model_running,
                       algorithm_path,
                       threshold,
                       features_list)

    @staticmethod
    def SVR_execution(test_data_path,
                      results_path,
                      similarity_score,
                      training_data_path,
                      save_model,
                      new_model_running,
                      algorithm_path,
                      threshold,
                      features_list):
        """
        executes Support Vector Regression algorithm
        :param test_data_path: path of test data set directory
        :param results_path: path of results  directory
        :param similarity_score: similarity functions
        :param training_data_path: path of train data set directory
        :param save_model: Indicator for saving the model or not
        :param new_model_running: Indicator whether the current flow is new model creation or not
        :param algorithm_path: path of existing model directory
        :param threshold: which was calculated in an existing model
        :param features_list: all the features in the test data set
        :return: results after model prediction
        """

        # Run SVR function
        run_svr_model(training_data_path,
                      test_data_path,
                      results_path,
                      similarity_score,
                      save_model,
                      new_model_running,
                      algorithm_path,
                      threshold,
                      features_list)

    @staticmethod
    def Random_Forest_execution(test_data_path,
                                results_path,
                                similarity_score,
                                training_data_path,
                                save_model,
                                new_model_running,
                                algorithm_path,
                                threshold,
                                features_list):
        """
        executes Random forest algorithm
        :param test_data_path: path of test data set directory
        :param results_path: path of results  directory
        :param similarity_score: similarity functions
        :param training_data_path: path of train data set directory
        :param save_model: Indicator for saving the model or not
        :param new_model_running: Indicator whether the current flow is new model creation or not
        :param algorithm_path: path of existing model directory
        :param threshold: which was calculated in an existing model
        :param features_list: all the features in the test data set
        :return: results after model prediction
        """

        # Run Random forest execution function
        run_random_forest_model(training_data_path,
                                test_data_path,
                                results_path,
                                similarity_score,
                                save_model,
                                new_model_running,
                                algorithm_path,
                                threshold,
                                features_list)

    @staticmethod
    def Linear_Regression_execution(test_data_path,
                                    results_path,
                                    similarity_score,
                                    training_data_path,
                                    save_model,
                                    new_model_running,
                                    algorithm_path,
                                    threshold,
                                    features_list):
        """
        executes Linear Regression algorithm
        :param test_data_path: path of test data set directory
        :param results_path: path of results  directory
        :param similarity_score: similarity functions
        :param training_data_path: path of train data set directory
        :param save_model: Indicator for saving the model or not
        :param new_model_running: Indicator whether the current flow is new model creation or not
        :param algorithm_path: path of existing model directory
        :param threshold: which was calculated in an existing model
        :param features_list: all the features in the test data set
        :return: results after model prediction
        """

        # Run Linear Regression execution function
        run_linear_regression_model(training_data_path,
                                    test_data_path,
                                    results_path,
                                    similarity_score,
                                    save_model,
                                    new_model_running,
                                    algorithm_path,
                                    threshold,
                                    features_list)

    @staticmethod
    def get_algorithm_execution_function(algorithm_name):
        """
        switch to get the execution function for a given algorithm
        :param algorithm_name:
        :return: execution function suitable to input algorithm
        """

        algorithms = load_anomaly_detection_list()

        # Switch to get the suitable execute function for a given algorithm
        switcher = {
            algorithms[0]: ModelsExecution.LSTM_execution,
            algorithms[1]: ModelsExecution.SVR_execution,
            algorithms[2]: ModelsExecution.Linear_Regression_execution,
            algorithms[3]: ModelsExecution.Random_Forest_execution
        }

        return switcher.get(algorithm_name, None)
