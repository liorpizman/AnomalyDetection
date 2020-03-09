from utils.shared.input_settings import input_settings
from utils.shared.models_execution import models_execution


class model_controller:

    def __init__(self, gui_controller):
        self.gui_controller = gui_controller

    def set_training_data_path(self, dir):
        input_settings.set_training_data_path(dir)

    def get_training_data_path(self):
        return input_settings.get_training_data_path()

    def set_test_data_path(self, dir):
        input_settings.set_test_data_path(dir)

    def get_test_data_path(self):
        return input_settings.get_test_data_path()

    def set_results_path(self, dir):
        input_settings.set_results_path(dir)

    def get_results_path(self):
        return input_settings.set_results_path()

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        input_settings.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def remove_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        input_settings.remove_algorithm_parameters(algorithm_name, algorithm_parameters)

    def set_similarity_score(self, similarity_list):
        input_settings.set_similarity_score(similarity_list)

    def set_saving_model(self, save_model):
        input_settings.set_saving_model(save_model)

    def run_models(self):
        models_execution.run_models()

    def set_new_model_running(self, new_model_running):
        input_settings.set_new_model_running(new_model_running)

    def set_existing_algorithms(self, algorithms_dict):
        input_settings.set_existing_algorithms(algorithms_dict)

    def set_existing_algorithms_threshold(self, threshold):
        input_settings.set_existing_algorithms_threshold(threshold)

    def set_features_columns_options(self):
        input_settings.set_features_columns_options()

    def get_features_columns_options(self):
        return input_settings.get_features_columns_options()

    def set_users_selected_features(self, features_list):
        input_settings.set_users_selected_features(features_list)
