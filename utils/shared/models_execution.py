from lstm_execution import run_model
from utils.shared.input_settings import input_settings


class models_execution:


    @staticmethod
    def run_models():
        # do it in function
        algorithms = input_settings.get_algorithms()
        similarity_score = input_settings.get_similarity()
        training_data_path = input_settings.get_training_data_path()
        test_data_path = input_settings.get_test_data_path()
        results_path = input_settings.get_results_path()
        save_model = input_settings.get_saving_model()

        for algorithm in algorithms:
            model_execution_function = getattr(models_execution, algorithm+"_execution")
            model_execution_function(training_data_path,
                                     test_data_path,
                                     results_path,
                                     similarity_score,
                                     save_model)

    @staticmethod
    def LSTM_execution(training_data_path,
                       test_data_path,
                       results_path,
                       similarity_score,
                       save_model):

        run_model(training_data_path,
                  test_data_path,
                  results_path,
                  similarity_score,
                  save_model)
