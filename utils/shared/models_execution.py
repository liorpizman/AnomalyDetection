from lstm_execution import run_model
from utils.shared.input_settings import input_settings


class models_execution:

    @classmethod
    def get_new_model_parameters(cls):
        return (input_settings.get_training_data_path(),
                input_settings.get_saving_model(),
                input_settings.get_algorithms(),
                None,)

    @classmethod
    def get_load_model_parameters(cls):
        return (None,
                False,
                input_settings.get_existing_algorithms(),
                input_settings.get_existing_algorithms_threshold(),)


    @classmethod
    def get_parameters(cls):
        return( input_settings.get_similarity(),
                input_settings.get_test_data_path(),
                input_settings.get_results_path(),
                input_settings.get_new_model_running(),
                input_settings.get_users_selected_features(),)

    @staticmethod
    def run_models():
        similarity_score,test_data_path ,results_path,new_model_running,features_list = models_execution.get_parameters()

        if new_model_running:
            training_data_path ,save_model ,algorithms ,threshold  = models_execution.get_new_model_parameters()
        else:
            training_data_path ,save_model ,algorithms ,threshold  = models_execution.get_load_model_parameters()

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
                                     threshold,
                                     features_list)

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
        run_model(training_data_path,
                  test_data_path,
                  results_path,
                  similarity_score,
                  save_model,
                  new_model_running,
                  algorithm_path,
                  threshold,
                  features_list)
