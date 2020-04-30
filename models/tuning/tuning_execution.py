'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Switcher between tuning models execution in the system
'''
from models.tuning.lstm_model_tuning import run_tuning as run_LSTM_tuning
from models.tuning.sklearn_model_tuning import run_tuning as run_sklearn_tuning
from utils.input_settings import InputSettings


class TuningExecution:
    """
    A Class used to execute different tuning of machine learning algorithms dynamically

    Methods
    -------
    run_tuning()
            Description | Executes the algorithm tuning function

    LSTM_tuning_execution(input_path,input_features,target_features,window_size,results_path)
            Description | Executes Long short-term memory tuning

    sklearn_tuning_execution(input_path,input_features,target_features,window_size,results_path , algorithm)
            Description | Executes sklearn model tuning

    """

    @classmethod
    def get_parameters(cls):
        """
        get tuning parameters
        :return: tuning flows parameters
        """

        return (
            InputSettings.get_tune_model_input_path(),
            InputSettings.get_tune_flow_input_features(),
            InputSettings.get_tune_flow_target_features(),
            InputSettings.get_tune_flow_window_size(),
            InputSettings.get_tune_flow_algorithm(),
            InputSettings.get_tune_model_results_path()
        )

    @staticmethod
    def run_tuning():
        """
        executes all the algorithms which were chosen - suitable for both flows
        :return:
        """

        input_path, input_features, target_features, window_size, algorithm, results_path = TuningExecution.get_parameters()

        if algorithm == "LSTM":
            TuningExecution.LSTM_tuning_execution(input_path,
                                                  input_features,
                                                  target_features,
                                                  window_size,
                                                  results_path)
        else:
            TuningExecution.sklearn_tuning_execution(input_path,
                                                     input_features,
                                                     target_features,
                                                     window_size,
                                                     results_path,
                                                     algorithm)

    @staticmethod
    def LSTM_tuning_execution(input_path, input_features, target_features, window_size, results_path):
        """
        executes LSTM algorithm tuning

        :param input_path: file input path
        :param input_features: input features
        :param target_features: target features
        :param window_size: window size
        :param results_path: results path
        :return:
        """

        run_LSTM_tuning(input_path, input_features, target_features, window_size, results_path)

        pass

    @staticmethod
    def sklearn_tuning_execution(input_path, input_features, target_features, window_size, results_path, algorithm):
        """
        executes sklearn algorithm tuning

        :param input_path: file input path
        :param input_features: input features
        :param target_features: target features
        :param window_size: window size
        :param results_path: results path
        :param algorithm: algorithm name
        :return:
        """

        run_sklearn_tuning(input_path, input_features, target_features, window_size, results_path, algorithm)
