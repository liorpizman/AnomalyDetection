from lstm_execution import run_model
from utils.shared.input_settings import input_settings


class models_execution:

    @staticmethod
    def run_models():
        similarity_score = input_settings.get_similarity()
        test_data_path = input_settings.get_test_data_path()
        results_path = input_settings.get_results_path()

        new_model_running = input_settings.get_new_model_running()
        if new_model_running:
            training_data_path = input_settings.get_training_data_path()
            save_model = input_settings.get_saving_model()
            algorithms = input_settings.get_algorithms()
            threshold = None
        else:
            training_data_path = None
            save_model = False
            algorithms = input_settings.get_existing_algorithms()
            threshold = input_settings.get_existing_algorithms_threshold()

        for algorithm in algorithms:
            if new_model_running:
                algorithm_path = None
            else:
                algorithm_path = input_settings.get_existing_algorithm_path(algorithm)
            model_execution_function = getattr(models_execution, algorithm + "_execution")
            model_execution_function(test_data_path,
                                     results_path,
                                     similarity_score,
                                     training_data_path,
                                     save_model,
                                     new_model_running,
                                     algorithm_path,
                                     threshold)

    @staticmethod
    def LSTM_execution(test_data_path,
                       results_path,
                       similarity_score,
                       training_data_path,
                       save_model,
                       new_model_running,
                       algorithm_path,
                       threshold):
        run_model(training_data_path,
                  test_data_path,
                  results_path,
                  similarity_score,
                  save_model,
                  new_model_running,
                  algorithm_path,
                  threshold)
