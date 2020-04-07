from utils.input_settings import InputSettings
from utils.models_execution import ModelsExecution


class ModelController:

    def __init__(self, gui_controller):
        self.gui_controller = gui_controller

    def set_training_data_path(self, dir):
        InputSettings.set_training_data_path(dir)

    def get_training_data_path(self):
        return InputSettings.get_training_data_path()

    def set_test_data_path(self, dir):
        InputSettings.set_test_data_path(dir)

    def get_test_data_path(self):
        return InputSettings.get_test_data_path()

    def set_results_path(self, dir):
        InputSettings.set_results_path(dir)

    def get_results_path(self):
        return InputSettings.set_results_path()

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        InputSettings.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def remove_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        InputSettings.remove_algorithm_parameters(algorithm_name, algorithm_parameters)

    def set_similarity_score(self, similarity_list):
        InputSettings.set_similarity_score(similarity_list)

    def set_saving_model(self, save_model):
        InputSettings.set_saving_model(save_model)

    def run_models(self):
        ModelsExecution.run_models()

    def set_new_model_running(self, new_model_running):
        InputSettings.set_new_model_running(new_model_running)

    def set_existing_algorithms(self, algorithms_dict):
        InputSettings.set_existing_algorithms(algorithms_dict)

    def set_features_columns_options(self):
        InputSettings.set_features_columns_options()

    def get_features_columns_options(self):
        return InputSettings.get_features_columns_options()

    def set_users_selected_features(self, algorithm_name, features_list):
        InputSettings.set_users_selected_features(algorithm_name, features_list)

    def add_new_thread(self, new_thread):
        InputSettings.add_new_thread(new_thread)

    def get_existing_thread(self):
        return InputSettings.get_existing_thread()

    def get_new_model_running(self):
        return InputSettings.get_new_model_running()

    def get_algorithms(self):
        return InputSettings.get_algorithms()

    def remove_algorithm(self, algorithm_name):
        InputSettings.remove_algorithm(algorithm_name)

    def remove_algorithms(self, algorithm_name):
        InputSettings.remove_algorithms(algorithm_name)

    def set_results_selected_algorithm(self, selected_algorithm):
        InputSettings.set_results_selected_algorithm(selected_algorithm)

    def set_results_selected_flight_route(self, selected_flight_route):
        InputSettings.set_results_selected_flight_route(selected_flight_route)

    def get_results_selected_algorithm(self):
        return InputSettings.get_results_selected_algorithm()

    def get_results_selected_flight_route(self):
        return InputSettings.get_results_selected_flight_route()
