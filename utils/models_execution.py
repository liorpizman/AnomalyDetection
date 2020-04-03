from gui.shared.helper_methods import read_json_file, get_model_path, load_anomaly_detection_list
from models.lstm.lstm_execution import run_model as run_lstm_model
from models.svr.svr_execution import run_model as run_svr_model
from models.random_forest.random_forest_execution import run_model as run_random_forest_model
from utils.input_settings import InputSettings
from utils.helper_methods import get_subdirectories


class ModelsExecution:

    @classmethod
    def get_new_model_parameters(cls):
        return (InputSettings.get_training_data_path(),
                InputSettings.get_saving_model(),
                InputSettings.get_algorithms(),
                None,
                InputSettings.get_users_selected_features(),)

    @classmethod
    def get_load_model_parameters(cls):
        return (None,
                False,
                InputSettings.get_existing_algorithms(),
                InputSettings.get_existing_algorithms_threshold(),)

    @classmethod
    def get_parameters(cls):
        return (InputSettings.get_similarity(),
                InputSettings.get_test_data_path(),
                InputSettings.get_results_path(),
                InputSettings.get_new_model_running(),)

    @staticmethod
    def run_models():
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
                algorithm_features_list = read_json_file(f'{algorithm_path}/model_data.json')['features']
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
    def get_algorithm_execution_function(algorithm_name):
        algorithms = load_anomaly_detection_list()
        switcher = {
            algorithms[0]: ModelsExecution.LSTM_execution,
            algorithms[1]: ModelsExecution.SVR_execution,
            # algorithms[2]: ModelsExecution.KNN_execution,
            algorithms[3]: ModelsExecution.Random_Forest_execution
        }
        return switcher.get(algorithm_name, None)
