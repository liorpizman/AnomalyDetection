#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Sklearn models tuning class
'''
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from gui.algorithm_frame_options.shared.helper_methods import load_algorithm_constants
from models.data_preprocessing.data_cleaning import clean_data
from models.data_preprocessing.data_normalization import normalize_data
from models.time_series_model.time_series_estimator import TimeSeriesRegressor, cascade_cv
from utils.helper_methods import get_current_time, plot_prediction_performance


def train_test_plot(pred_train, y_train, title, results_path, target_features):
    # plt.subplot(1, 2, 1)
    for i, column in enumerate(target_features):
        feature_title = str(title) + " for " + str(column)
        plt.plot(pred_train[i], 'r', label="Predicted")
        plt.plot(pred_train[i], 'r', label="Predicted")
        plt.plot(y_train[i], 'b--', label="Actual")
        # nprev: because the first predicted point needed n_prev steps of data
        plt.title("Test performance of " + feature_title)
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 6)

        plt.savefig(f'{results_path}/{feature_title}.png')

        plt.clf()

        plt.show()


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


def get_suitable_yaml_file(model_name):
    """
    get the yaml file according to algorithm name
    :param model_name: the name of the algorithm
    :return: yaml file
    """

    switcher = {
        "SVR": "svr_params.yaml",
        "MLP": "mlp_params.yaml",
        "Random Forest": "random_forest_params.yaml",
    }
    return switcher.get(model_name, None)


def get_suitable_SVR_params(model_params):
    """
    get suitable svr params option
    :return: svr params
    """

    for key in model_params.keys():
        if key == "base_estimator__estimator__epsilon":
            for index in range(len(model_params[key])):
                model_params[key][index] = float(model_params[key][index])

    return model_params


def get_suitable_Random_Forest_params(model_params):
    """
    get suitable random forest params option
    :return: Random Forest params
    """

    for key in model_params.keys():
        if key == "base_estimator__estimator__n_estimators":
            for index in range(len(model_params[key])):
                model_params[key][index] = int(model_params[key][index])
        elif key == "base_estimator__estimator__random_state":
            for index in range(len(model_params[key])):
                model_params[key][index] = int(model_params[key][index])

    # del model_params["base_estimator__estimator__n_estimators"]
    # del model_params["base_estimator__estimator__random_state"]
    # del model_params["base_estimator__estimator__max_features"]

    return model_params


def get_suitable_MLP_params(model_params):
    """
    get suitable mlp params option
    :return: MLP params
    """

    for key in model_params.keys():
        if key == "base_estimator__estimator__hidden_layer_sizes":
            for index in range(len(model_params[key])):
                model_params[key][index] = eval(model_params[key][index])
        elif key == "base_estimator__estimator__alpha":
            for index in range(len(model_params[key])):
                model_params[key][index] = float(model_params[key][index])
        elif key == "base_estimator__estimator__random_state":
            for index in range(len(model_params[key])):
                model_params[key][index] = int(model_params[key][index])

    return model_params


def get_suitable_params_values(model_name, model_params):
    """

    :param model_name: algorithm name
    :param model_params: params dict
    :return:
    """

    switcher = {
        "SVR": lambda model_params: get_suitable_SVR_params(model_params),
        "MLP": lambda model_params: get_suitable_MLP_params(model_params),
        "Random Forest": lambda model_params: get_suitable_Random_Forest_params(model_params)
    }
    return switcher.get(model_name, None)(model_params)


def get_params_from_yaml(model_name, yaml_filename):
    """
    get model's params from yaml file
    :param model_name: algorithm name
    :param yaml_filename: yaml filename
    :return:
    """

    yaml_params = load_algorithm_constants(yaml_filename)
    parameters_lists_keys = list(yaml_params.keys())

    # Set values for frame construction
    values_lists = []
    for key in parameters_lists_keys:
        values_lists.append(yaml_params.get(key))

    # Pop keys of each list
    values_lists.pop(0)  # remove first element
    tmp_params_keys = values_lists.pop(0)  # remove first element

    for i in range(2):  # remove threshold and window size elements
        values_lists.pop()  # threshold element
        tmp_params_keys.pop()

    params_keys_lists = []
    for param_key in tmp_params_keys:
        params_keys_lists.append("base_estimator__estimator__" + param_key)
        # params_keys_lists.append(param_key)

    params_values = convert_string_to_boolean(values_lists)

    params_dict = dict(zip(params_keys_lists, params_values))

    return get_suitable_params_values(model_name, params_dict)


def get_SVR_params():
    """
    get svr params option
    :return: svr params
    """

    current_yaml_name = get_suitable_yaml_file("SVR")

    return get_params_from_yaml("SVR", current_yaml_name)


def get_Random_Forest_params():
    """
    get random forest params option
    :return: Random Forest params
    """

    current_yaml_name = get_suitable_yaml_file("Random Forest")

    return get_params_from_yaml("Random Forest", current_yaml_name)


def get_MLP_params():
    """
    get mlp params option
    :return: MLP params
    """

    current_yaml_name = get_suitable_yaml_file("MLP")

    params = get_params_from_yaml("MLP", current_yaml_name)

    return params


def get_model(model_name):
    """
    get the model
    :param model_name: model name
    :return: sklearn model
    """

    switcher = {
        "SVR": lambda: MultiOutputRegressor(SVR()),
        "MLP": lambda: MultiOutputRegressor(MLPRegressor(shuffle=False)),
        "Random Forest": lambda: MultiOutputRegressor(RandomForestRegressor())
    }
    return switcher.get(model_name, lambda: MultiOutputRegressor(SVR()))()


def get_model_params(model_name):
    """
    get the model params
    :param model_name: model name
    :return: model params
    """

    switcher = {
        "SVR": lambda: get_SVR_params(),
        "MLP": lambda: get_MLP_params(),
        "Random Forest": lambda: get_Random_Forest_params()
    }
    return [switcher.get(model_name, None)()]


def model_tuning(file_path, input_features, target_features, window_size, scaler, results_path, model_name):
    """
    model's tuning process by using GridSearchCV
    :param file_path: data file path
    :param input_features: the list of features which the user chose for the train
    :param target_features: the list of features which the user chose for the test
    :param window_size: window size variable
    :param scaler: scaler name
    :param results_path: results path
    :param model_name: model name
    :return: model name , best models params
    """

    df_train = pd.read_csv(f'{file_path}')

    input_df_train = df_train[input_features]
    target_df_train = df_train[target_features]

    # Step 1 : Clean train data set
    input_df_train = clean_data(input_df_train)

    target_df_train = clean_data(target_df_train)

    # Step 2: Normalize the data

    X = normalize_data(data=input_df_train,
                       scaler=scaler)[0]

    Y = normalize_data(data=target_df_train,
                       scaler=scaler)[0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    model = get_model(model_name)

    model_grid_params = get_model_params(model_name)

    tsr = TimeSeriesRegressor(model, n_prev=window_size)

    grid_search = GridSearchCV(tsr, model_grid_params)

    grid_search.fit(X_train, Y_train)
    prediction = grid_search.predict(X_test)

    plot_title = "Optimized Time Series " + model_name + " model"
    print(str(model_name) + " " + str(grid_search.best_params_))
    current_time = get_current_time()
    file_name = str(current_time) + "-" + str(model_name) + "-model_data.json"
    data = {}
    data['model'] = model_name
    data["input_features"] = input_features
    data["target_features"] = target_features
    data['params'] = grid_search.best_params_
    data['score'] = grid_search.best_score_

    with open(f'{results_path}/{file_name}', 'w') as outfile:
        json.dump(data, outfile)

    # train_test_plot(pred_train=prediction,
    #                 y_train=tsr._preprocess(Y_test, Y_test)[1],
    #                 title=plot_title,
    #                 results_path=results_path,
    #                 target_features=target_features
    #                 )

    Y_test_preprocessed = tsr._preprocess(X_test, Y_test)[1]

    for i, target_feature in enumerate(target_features):
        title = "Grid search test performance of " + model_name + " for window size: " + \
                str(window_size) + " and " + target_feature + " feature"
        plot_prediction_performance(Y_train=Y_test_preprocessed[:, i],
                                    X_pred=prediction[:, i],
                                    results_path=results_path,
                                    title=title)

    return data['params'], data['score']


def run_tuning(file_path, input_features, target_features, window_size, results_path, algorithm):
    results = dict()
    for window in window_size:
        results[window] = model_tuning(file_path,
                                       input_features,
                                       target_features,
                                       int(window),
                                       "min_max",
                                       results_path,
                                       algorithm)

    # return results
