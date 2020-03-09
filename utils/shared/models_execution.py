from lstm_execution import run_model
from utils.shared.input_settings import InputSettings


class ModelsExecution:

    @staticmethod
    def run_models():
        similarity_score = InputSettings.get_similarity()
        test_data_path = InputSettings.get_test_data_path()
        results_path = InputSettings.get_results_path()

        new_model_running = InputSettings.get_new_model_running()
        if new_model_running:
            training_data_path = InputSettings.get_training_data_path()
            save_model = InputSettings.get_saving_model()
            algorithms = InputSettings.get_algorithms()
            threshold = None
        else:
            training_data_path = None
            save_model = False
            algorithms = InputSettings.get_existing_algorithms()
            threshold = InputSettings.get_existing_algorithms_threshold()

        for algorithm in algorithms:
            if new_model_running:
                algorithm_path = None
            else:
                algorithm_path = InputSettings.get_existing_algorithm_path(algorithm)
            model_execution_function = getattr(ModelsExecution, algorithm + "_execution")
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
