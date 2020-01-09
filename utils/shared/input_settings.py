from utils.shared.lstm_hyper_parameters import lstm_hyper_parameters


class input_settings:

    TRAINING_DATA_PATH = ""
    TEST_DATA_PATH = ""
    RESULTS_DATA_PATH = ""
    ALGORITHMS = set()
    SIMILARITY_SCORES = set()
    SAVE_MODEL = False

    @staticmethod
    def set_training_data_path(path):
        input_settings.TRAINING_DATA_PATH = path

    @staticmethod
    def get_training_data_path():
        return input_settings.TRAINING_DATA_PATH

    @staticmethod
    def set_test_data_path(path):
        input_settings.TEST_DATA_PATH = path

    @staticmethod
    def get_test_data_path():
        return input_settings.TEST_DATA_PATH

    @staticmethod
    def set_results_path(path):
        input_settings.RESULTS_DATA_PATH = path

    @staticmethod
    def get_results_path():
        return input_settings.RESULTS_DATA_PATH

    @staticmethod
    def get_algorithms():
        return input_settings.ALGORITHMS

    @staticmethod
    def get_similarity():
        return input_settings.SIMILARITY_SCORES

    @staticmethod
    def set_algorithm_parameters(algorithm_name,algorithm_parameters):
        algorithm_setting_function = getattr(input_settings, "set_"+algorithm_name)
        algorithm_setting_function(algorithm_parameters)

    @staticmethod
    def set_LSTM(algorithm_parameters):
        input_settings.ALGORITHMS.add("LSTM")
        for param in algorithm_parameters:
            lstm_setting_function = getattr(lstm_hyper_parameters, "set_" + param)
            lstm_setting_function(algorithm_parameters[param])

    @staticmethod
    def remove_algorithm_parameters(algorithm_name, algorithm_parameters):
        algorithm_remove_function = getattr(input_settings, "remove_"+algorithm_name)
        algorithm_remove_function(algorithm_parameters)

    @staticmethod
    def remove_LSTM(algorithm_parameters):
        if "LSTM" not in input_settings.ALGORITHMS:
            return
        input_settings.ALGORITHMS.remove("LSTM")
        for param in algorithm_parameters:
            lstm_setting_function = getattr(lstm_hyper_parameters, "remove_" + param)
            lstm_setting_function(algorithm_parameters[param])

    @staticmethod
    def set_similarity_score(similarity_list):
        input_settings.SIMILARITY_SCORES = similarity_list

    @staticmethod
    def set_saving_model(save_model):
        input_settings.SAVE_MODEL = save_model

    @staticmethod
    def get_saving_model():
        return input_settings.SAVE_MODEL





