#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
SVR train and prediction execution function
'''

import pickle
import json
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from models.data_preprocessing.data_cleaning import clean_data
from models.data_preprocessing.data_normalization import normalize_data
from models.svr.svr_hyper_parameters import svr_hyper_parameters
from models.time_series_model.time_series_estimator import TimeSeriesRegressor
from utils.constants import ATTACK_COLUMN
from utils.routes import *
from utils.helper_methods import get_threshold, report_results, get_method_scores, get_subdirectories, \
    create_directories, get_current_time, \
    plot_reconstruction_error_scatter, get_attack_boundaries, anomaly_score
from collections import defaultdict


def get_svr_new_model_parameters():
    """
    Get SVR hyper parameters
    :return: SVR hyper parameters
    """

    return (
        svr_hyper_parameters.get_kernel(),
        svr_hyper_parameters.get_gamma(),
        svr_hyper_parameters.get_epsilon(),
        svr_hyper_parameters.get_threshold(),
        2
    )


def get_svr_model(kernel, gamma, epsilon):
    """
    Get SVR model
    :return: SVR model
    """

    return MultiOutputRegressor(SVR(kernel=kernel,
                                    gamma=gamma,
                                    epsilon=epsilon))


def run_model(training_data_path, test_data_path, results_path, similarity_score, save_model, new_model_running,
              algorithm_path, threshold, features_list, scalar_path):
    """
    Run SVR model process
    :param training_data_path: train data set directory path
    :param test_data_path: test data set directory path
    :param results_path: results directory path
    :param similarity_score: chosen similarity functions
    :param save_model: indicator whether the user want to save the model or not
    :param new_model_running: indicator whether we are in new model creation flow or not
    :param algorithm_path: path of existing algorithm
    :param threshold: saved threshold for load model flow
    :param features_list:  saved chosen features for load model flow
    :param scalar_path: path of existing scalar directory
    :return:  reported results for SVR execution
    """

    # Choose between new model creation flow and load existing model flow
    if new_model_running:
        kernel, gamma, epsilon, threshold, window_size = get_svr_new_model_parameters()
    else:
        svr_model = pickle.load(open(algorithm_path, 'rb'))
        scalar = pickle.load(open(scalar_path, 'rb'))
        X_train = None

    FLIGHT_ROUTES = get_subdirectories(test_data_path)

    current_time = get_current_time()

    create_directories(f'{results_path}/svr/{current_time}')

    # Create sub directories for each similarity function
    for similarity in similarity_score:
        create_directories(f'{results_path}/svr/{current_time}/{similarity}')

    # Train the model for each flight route
    for flight_route in FLIGHT_ROUTES:

        # Execute training for new model flow
        if new_model_running:
            svr_model, scalar, X_train = execute_train(flight_route,
                                                       training_data_path=training_data_path,
                                                       kernel=kernel,
                                                       gamma=gamma,
                                                       epsilon=epsilon,
                                                       features_list=features_list,
                                                       window_size=window_size)

        # Get results for each similarity function
        for similarity in similarity_score:
            current_results_path = f'{results_path}/svr/{current_time}/{similarity}/{flight_route}'
            create_directories(current_results_path)
            tpr_scores, fpr_scores, acc_scores, delay_scores = execute_predict(flight_route,
                                                                               test_data_path=test_data_path,
                                                                               similarity_score=similarity,
                                                                               threshold=threshold,
                                                                               svr_model=svr_model,
                                                                               scalar=scalar,
                                                                               results_path=current_results_path,
                                                                               add_plots=True,
                                                                               run_new_model=new_model_running,
                                                                               X_train=X_train,
                                                                               features_list=features_list,
                                                                               save_model=save_model)

            df = pd.DataFrame(tpr_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_tpr.csv', index=False)

            df = pd.DataFrame(fpr_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_fpr.csv', index=False)

            df = pd.DataFrame(acc_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_acc.csv', index=False)

            df = pd.DataFrame(delay_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_delay.csv', index=False)

    algorithm_name = "SVR"

    # Report results for training data to csv files
    for similarity in similarity_score:
        report_results(f'{results_path}/svr/{current_time}/{similarity}',
                       test_data_path,
                       FLIGHT_ROUTES,
                       algorithm_name,
                       similarity)


def execute_train(flight_route,
                  training_data_path=None,
                  kernel=None,
                  gamma=None,
                  epsilon=None,
                  features_list=None,
                  window_size=1):
    """
    Execute train function for a specific flight route
    :param flight_route: current flight route we should train on
    :param training_data_path: the path of training data directory
    :param kernel: kernel string values
    :param gamma: gamma string value
    :param epsilon: epsilon float value
    :param features_list: the list of features which the user chose
    :return: svr model, normalization scalar, X_train data frame
    """

    df_train = pd.read_csv(f'{training_data_path}/{flight_route}/without_anom.csv')

    df_train = df_train[features_list]

    # Step 1 : Clean train data set
    df_train = clean_data(df_train)

    # Step 2: Normalize the data
    X_train, scalar = normalize_data(data=df_train,
                                     scaler="max_abs")

    # Get the model which is created by user's parameters
    svr_model = get_svr_model(kernel=kernel,
                              gamma=gamma,
                              epsilon=epsilon)

    tsr = TimeSeriesRegressor(svr_model, n_prev=window_size)
    tsr.fit(X_train)

    return tsr, scalar, X_train


def execute_predict(flight_route,
                    test_data_path=None,
                    similarity_score=None,
                    threshold=None,
                    svr_model=None,
                    scalar=None,
                    results_path=None,
                    add_plots=True,
                    run_new_model=False,
                    X_train=None,
                    features_list=None,
                    save_model=False):
    """
    Execute predictions function for a specific flight route
    :param flight_route: current flight route we should train on
    :param test_data_path: the path of test data directory
    :param similarity_score: similarity function
    :param threshold: threshold from the train
    :param svr_model: SVR model
    :param scalar: normalization scalar
    :param results_path: the path of results directory
    :param add_plots: indicator whether to add plots or not
    :param run_new_model: indicator whether current flow is new model creation or not
    :param X_train: data frame
    :param features_list: the list of features which the user chose
    :param save_model: indicator whether the user want to save the model or not
    :return: tpr scores, fpr scores, acc scores, delay scores
    """

    tpr_scores = defaultdict(list)
    fpr_scores = defaultdict(list)
    acc_scores = defaultdict(list)
    delay_scores = defaultdict(list)

    # Set a threshold in new model creation flow
    if run_new_model:
        threshold = predict_train_set(svr_model,
                                      X_train,
                                      save_model,
                                      add_plots,
                                      threshold,
                                      features_list,
                                      results_path,
                                      flight_route,
                                      similarity_score,
                                      scalar)

    flight_dir = os.path.join(test_data_path, flight_route)
    ATTACKS = get_subdirectories(flight_dir)

    # Iterate over all attacks in order to find anomalies
    for attack in ATTACKS:
        for flight_csv in os.listdir(f'{test_data_path}/{flight_route}/{attack}'):

            df_test_source = pd.read_csv(f'{test_data_path}/{flight_route}/{attack}/{flight_csv}')

            Y_test_labels = df_test_source[[ATTACK_COLUMN]].values
            Y_test_target = svr_model._preprocess(Y_test_labels, Y_test_labels)[1]

            df_test = df_test_source[features_list]

            # Step 1 : Clean test data set
            df_test = clean_data(df_test)

            # Step 2: Normalize the data
            X_test = scalar.transform(df_test)

            X_pred = svr_model.predict(X_test)

            scores_test = []
            for i, pred in enumerate(X_pred):
                scores_test.append(anomaly_score(X_test[i], pred, similarity_score))

            # Add reconstruction error scatter if plots indicator is true
            if add_plots:
                plot_reconstruction_error_scatter(scores=scores_test,
                                                  labels=Y_test_target,
                                                  threshold=threshold,
                                                  plot_dir=results_path,
                                                  title=f'Outlier Score Testing for {flight_csv} in {flight_route}({attack})')

            predictions = [1 if x >= threshold else 0 for x in scores_test]

            # Add roc curve if plots indicator is true
            if add_plots:
                pass
                # plot_roc(y_true=Y_test,y_pred=predictions, plot_dir=results_path,title=f'ROC Curve - {flight_csv} in {flight_route}({attack})')

            attack_start, attack_end = get_attack_boundaries(df_test_source[ATTACK_COLUMN])

            method_scores = get_method_scores(predictions, attack_start, attack_end,
                                              add_window_size=False, window_size=None)

            tpr_scores[attack].append(method_scores[0])
            fpr_scores[attack].append(method_scores[1])
            acc_scores[attack].append(method_scores[2])
            delay_scores[attack].append(method_scores[3])

    return tpr_scores, fpr_scores, acc_scores, delay_scores


def predict_train_set(svr_model,
                      X_train,
                      save_model,
                      add_plots,
                      threshold,
                      features_list,
                      results_path,
                      flight_route,
                      similarity_score,
                      scalar):
    """
    Execute prediction on the train data set
    :param svr_model: SVR model
    :param X_train: data frame
    :param save_model: indicator whether the user want to save the model or not
    :param add_plots: indicator whether to add plots or not
    :param threshold: threshold from the train
    :param features_list: the list of features which the user chose
    :param results_path: the path of results directory
    :param flight_route: current flight route we are working on
    :param similarity_score: similarity function
    :param scalar: normalization scalar
    :return: threshold
    """

    X_pred = svr_model.predict(X_train)
    scores_train = []

    for i, pred in enumerate(X_pred):
        scores_train.append(anomaly_score(X_train[i], pred, similarity_score))

    # choose threshold for which <MODEL_THRESHOLD_FROM_TRAINING_PERCENT> % of training were lower
    threshold = get_threshold(scores_train, threshold)

    # Save created model if the indicator is true
    if save_model:
        data = {}
        data['features'] = features_list
        data['threshold'] = threshold
        with open(f'{results_path}/model_data.json', 'w') as outfile:
            json.dump(data, outfile)
        save_model_file_path = os.path.join(results_path, flight_route + "_model.pkl")
        with open(save_model_file_path, 'wb') as file:
            pickle.dump(svr_model, file)
        save_scalar_file_path = os.path.join(results_path, flight_route + "_scalar.pkl")
        with open(save_scalar_file_path, 'wb') as file:
            pickle.dump(scalar, file)

    return threshold
