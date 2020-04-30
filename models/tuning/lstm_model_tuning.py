import json
from statistics import mean

import pandas as pd
from sklearn.model_selection import train_test_split

from gui.algorithm_frame_options.shared.helper_methods import load_algorithm_constants
from models.data_preprocessing.data_normalization import normalize_data
from models.lstm.lstm_autoencoder import get_lstm_autoencoder_model
from utils.helper_methods import get_training_data_lstm, anomaly_score_multi, get_current_time


def convert_string_to_boolean(source_dict):
    """
    Converts string values of array to booleans in which boolean values are presented as a string
    :param source_dict: input value with non boolean values
    :return: transformed dictionary with boolean values
    """

    changes = {
        "True": True,
        "False": False
    }

    for key in range(len(source_dict)):
        source_dict[key] = [changes.get(x, x) for x in source_dict[key]]

    return source_dict


def convert_params_dict_to_list(model_params_dict):
    """
    convert params dict to list
    :param model_params_dict: model params dict
    :return: model params dict to list
    """
    configs = list()
    for encoding_dimension in model_params_dict["encoding_dimension"]:
        for activation in model_params_dict["activation"]:
            for loss in model_params_dict["loss"]:
                for optimizer in model_params_dict["optimizer"]:
                    for epoch in model_params_dict["epochs"]:
                        cfg = [
                            encoding_dimension,
                            activation,
                            loss,
                            optimizer,
                            epoch
                        ]
                        configs.append(cfg)

    return configs


def get_suitable_params_values(model_params):
    """
    get suitable random forest params option
    :return: svr params
    """

    del model_params["window_size"]
    for key in model_params.keys():
        if key == "encoding_dimension":
            for index in range(len(model_params[key])):
                model_params[key][index] = int(model_params[key][index])
        elif key == "epochs":
            for index in range(len(model_params[key])):
                model_params[key][index] = int(model_params[key][index])

    return convert_params_dict_to_list(model_params)


def get_params_from_yaml():
    """
    get model's params from yaml file
    :param model_name: algorithm name
    :return:
    """

    yaml_params = load_algorithm_constants("lstm_params.yaml")
    parameters_lists_keys = list(yaml_params.keys())

    # Set values for frame construction
    values_lists = []
    for key in parameters_lists_keys:
        values_lists.append(yaml_params.get(key))

    # Pop keys of each list
    values_lists.pop(0)  # remove first element
    tmp_params_keys = values_lists.pop(0)  # remove first element

    # remove threshold
    values_lists.pop()
    tmp_params_keys.pop()

    params_keys_lists = []
    for param_key in tmp_params_keys:
        params_keys_lists.append(param_key)

    params_values = convert_string_to_boolean(values_lists)

    params_dict = dict(zip(params_keys_lists, params_values))

    return get_suitable_params_values(params_dict)


def get_lstm_params_configurations():
    return get_params_from_yaml()


def model_tuning(file_path, input_features, target_features, window_size, scaler, results_path):
    """
    model's tuning process by using GridSearchCV
    :param model_name: model name
    :param file_path: data file  path
    :param input_features: the list of features which the user chose for the train
    :param target_features: the list of features which the user chose for the test
    :param window_size: window size variable
    :param scaler: scaler name
    :param results_path: results path
    :return: model name , best models params
    """

    df_train = pd.read_csv(f'{file_path}')

    input_df_train = df_train[input_features]
    target_df_train = df_train[target_features]

    X = normalize_data(data=input_df_train,
                       scaler=scaler)[0]

    Y = normalize_data(data=target_df_train,
                       scaler=scaler)[0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    assert len(X_train) == len(Y_train)
    assert len(X_test) == len(Y_test)

    X_train_preprocessed = get_training_data_lstm(X_train, window_size)
    X_test_preprocessed = get_training_data_lstm(X_test, window_size)

    Y_train_preprocessed = get_training_data_lstm(Y_train, window_size)
    Y_test_preprocessed = get_training_data_lstm(Y_test, window_size)

    params_configurations = get_lstm_params_configurations()

    total_scores = dict()

    for config in params_configurations:
        encoding_dimension, activation, loss, optimizer, epochs = config

        lstm_model = get_lstm_autoencoder_model(timesteps=window_size,
                                                input_features=input_df_train.shape[1],
                                                target_features=target_df_train.shape[1],
                                                encoding_dimension=encoding_dimension,
                                                activation=activation,
                                                loss=loss,
                                                optimizer=optimizer)
        lstm_model.fit(X_train_preprocessed, Y_train_preprocessed, epochs=epochs, verbose=0)

        X_test_pred = lstm_model.predict(X_test_preprocessed)

        scores = []
        for i, pred in enumerate(X_test_pred):
            scores.append(anomaly_score_multi(Y_test_preprocessed[i], pred, 'MSE'))

        total_scores[str(config)] = mean(scores)

    total_sorted = {k: v for k, v in sorted(total_scores.items(), key=lambda item: item[1])}

    best_config = list(total_sorted.items())[0][0]
    best_score = list(total_sorted.items())[0][1]
    print(best_config)
    print(best_score)

    current_time = get_current_time()
    file_name = str(current_time) + "-LSTM-model_data.json"
    data = {}
    data['model'] = 'LSTM'
    data["input_features"] = input_features
    data["target_features"] = target_features
    data["window_size"] = window_size
    data['params'] = best_config
    data['score'] = best_score

    with open(f'{results_path}/{file_name}', 'w') as outfile:
        json.dump(data, outfile)

    return data['params'], data['score']


def run_tuning(file_path, input_features, target_features, window_size, results_path):
    results = dict()
    for window in window_size:
        results[window] = model_tuning(file_path,
                                       input_features,
                                       target_features,
                                       int(window),
                                       "min_max",
                                       results_path)

    # return results
