#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Random forest train and prediction execution function
'''

import pickle
import json
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from models.data_preprocessing.data_cleaning import clean_data
from models.data_preprocessing.data_normalization import normalize_data
from models.random_forest.random_forest_hyper_parameters import random_forest_hyper_parameters
from models.time_series_model.time_series_estimator import TimeSeriesRegressor
from utils.constants import ATTACK_COLUMN
from utils.routes import *
from utils.helper_methods import get_threshold, report_results, get_method_scores, get_subdirectories, \
    create_directories, get_current_time, \
    plot_reconstruction_error_scatter, get_attack_boundaries, anomaly_score, plot_prediction_performance
from collections import defaultdict


def get_random_forest_new_model_parameters():
    """
    Get random forest hyper parameters
    :return:random forest hyper parameters
    """

    return (
        random_forest_hyper_parameters.get_n_estimators(),
        random_forest_hyper_parameters.get_criterion(),
        random_forest_hyper_parameters.get_max_features(),
        random_forest_hyper_parameters.get_random_state(),
        random_forest_hyper_parameters.get_threshold(),
        random_forest_hyper_parameters.get_window_size()
    )


def get_random_forest_parameters_dictionary():
    """
    Get random forest hyper parameters dictionary
    :return: random forest hyper parameters dictionary
    """

    parameters = dict()

    parameters["n_estimators"] = random_forest_hyper_parameters.get_n_estimators()
    parameters["criterion"] = random_forest_hyper_parameters.get_criterion()
    parameters["max_features"] = random_forest_hyper_parameters.get_max_features()
    parameters["random_state"] = random_forest_hyper_parameters.get_random_state()
    parameters["threshold percent"] = random_forest_hyper_parameters.get_threshold()
    parameters["window size"] = random_forest_hyper_parameters.get_window_size()

    return parameters


def get_random_forest_model(n_estimators, criterion, max_features, random_state):
    """
    Get random forest model
    :return: random forest model
    """

    return MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                                      criterion=criterion,
                                                      max_features=max_features,
                                                      random_state=random_state))


def run_model(training_data_path, test_data_path, results_path, similarity_score, save_model, new_model_running,
              algorithm_path, threshold, features_list, target_features_list, train_scaler_path, target_scaler_path):
    """
    Run Random forest model process
    :param training_data_path: train data set directory path
    :param test_data_path: test data set directory path
    :param results_path: results directory path
    :param similarity_score: chosen similarity functions
    :param save_model: indicator whether the user want to save the model or not
    :param new_model_running: indicator whether we are in new model creation flow or not
    :param algorithm_path: path of existing algorithm
    :param threshold: saved threshold for load model flow
    :param features_list:  the list of features which the user chose for the train
    :param target_features_list: all the features in the test data set for the target
    :param train_scaler_path: path of existing input train scaler directory
    :param target_scaler_path: path of existing input target scaler directory
    :return:  reported results for Random forest execution
    """

    # Choose between new model creation flow and load existing model flow
    if new_model_running:
        n_estimators, criterion, max_features, \
        random_state, threshold, window_size = get_random_forest_new_model_parameters()
    else:
        random_forest_model = pickle.load(open(algorithm_path, 'rb'))
        X_train_scaler = pickle.load(open(train_scaler_path, 'rb'))
        Y_train_scaler = pickle.load(open(target_scaler_path, 'rb'))
        window_size = random_forest_model.n_prev
        X_train = None
        Y_train = None

    FLIGHT_ROUTES = get_subdirectories(test_data_path)

    current_time = get_current_time()

    current_time_path = os.path.join(*[str(results_path), 'random_forest', str(current_time)])
    create_directories(f"{current_time_path}")

    # Create sub directories for each similarity function
    for similarity in similarity_score:
        similarity_path = os.path.join(*[str(current_time_path), str(similarity)])
        create_directories(f"{similarity_path}")

    # Train the model for each flight route
    for flight_route in FLIGHT_ROUTES:

        # Execute training for new model flow
        if new_model_running:
            random_forest_model, X_train_scaler, Y_train_scaler, X_train, Y_train = execute_train(flight_route,
                                                                                                  training_data_path=training_data_path,
                                                                                                  n_estimators=n_estimators,
                                                                                                  criterion=criterion,
                                                                                                  max_features=max_features,
                                                                                                  random_state=random_state,
                                                                                                  features_list=features_list,
                                                                                                  window_size=window_size,
                                                                                                  target_features_list=target_features_list)

        # Get results for each similarity function
        for similarity in similarity_score:
            current_results_path = os.path.join(*[str(current_time_path), str(similarity), str(flight_route)])
            create_directories(f"{current_results_path}")
            tpr_scores, fpr_scores, acc_scores, delay_scores, routes_duration = execute_predict(flight_route,
                                                                                                test_data_path=test_data_path,
                                                                                                similarity_score=similarity,
                                                                                                threshold=threshold,
                                                                                                random_forest_model=random_forest_model,
                                                                                                X_train_scaler=X_train_scaler,
                                                                                                results_path=current_results_path,
                                                                                                add_plots=True,
                                                                                                run_new_model=new_model_running,
                                                                                                X_train=X_train,
                                                                                                features_list=features_list,
                                                                                                target_features_list=target_features_list,
                                                                                                save_model=save_model,
                                                                                                Y_train_scaler=Y_train_scaler,
                                                                                                Y_train=Y_train,
                                                                                                window_size=window_size)

            df = pd.DataFrame(tpr_scores)
            tpr_path = os.path.join(*[str(current_results_path), str(flight_route) + '_tpr.csv'])
            df.to_csv(f"{tpr_path}", index=False)

            df = pd.DataFrame(fpr_scores)
            fpr_path = os.path.join(*[str(current_results_path), str(flight_route) + '_fpr.csv'])
            df.to_csv(f"{fpr_path}", index=False)

            df = pd.DataFrame(acc_scores)
            acc_path = os.path.join(*[str(current_results_path), str(flight_route) + '_acc.csv'])
            df.to_csv(f"{acc_path}", index=False)

            df = pd.DataFrame(delay_scores)
            delay_path = os.path.join(*[str(current_results_path), str(flight_route) + '_delay.csv'])
            df.to_csv(f"{delay_path}", index=False)

    algorithm_name = "Random Forest"

    # Report results for training data to csv files
    for similarity in similarity_score:
        report_similarity_path = os.path.join(*[str(results_path), 'random_forest', str(current_time), str(similarity)])
        report_results(f"{report_similarity_path}",
                       test_data_path,
                       FLIGHT_ROUTES,
                       algorithm_name,
                       similarity,
                       routes_duration)


def execute_train(flight_route,
                  training_data_path=None,
                  n_estimators=None,
                  criterion=None,
                  max_features=None,
                  random_state=None,
                  features_list=None,
                  window_size=1,
                  target_features_list=None):
    """
    Execute train function for a specific flight route
    :param flight_route: current flight route we should train on
    :param training_data_path: the path of training data directory
    :param n_estimators: n estimators value
    :param criterion: criterion variable
    :param max_features: max amount of features
    :param random_state: random state value
    :param features_list: the list of features which the user chose for the train
    :param window_size: window size for each instance in training
    :param target_features_list: the list of features which the user chose for the target
    :return: random forest model, normalization input train scalar,normalization input target scalar, X_train data frame,Y_train data frame
    """

    without_anomaly_path = os.path.join(*[str(training_data_path), str(flight_route), 'without_anom.csv'])
    df_train = pd.read_csv(f"{without_anomaly_path}")

    input_df_train = df_train[features_list]
    target_df_train = df_train[target_features_list]

    # Step 1 : Clean train data set
    input_df_train = clean_data(input_df_train)

    target_df_train = clean_data(target_df_train)

    # Step 2: Normalize the data
    X_train, X_train_scaler = normalize_data(data=input_df_train,
                                             scaler="min_max")

    Y_train, Y_train_scaler = normalize_data(data=target_df_train,  # target data
                                             scaler="min_max")

    # Get the model which is created by user's parameters
    random_forest_model = get_random_forest_model(n_estimators=n_estimators,
                                                  criterion=criterion,
                                                  max_features=max_features,
                                                  random_state=random_state)
    tsr = TimeSeriesRegressor(random_forest_model, n_prev=window_size)
    tsr.fit(X_train, Y_train)

    return tsr, X_train_scaler, Y_train_scaler, X_train, Y_train


def execute_predict(flight_route,
                    test_data_path=None,
                    similarity_score=None,
                    threshold=None,
                    random_forest_model=None,
                    X_train_scaler=None,
                    results_path=None,
                    add_plots=True,
                    run_new_model=False,
                    X_train=None,
                    features_list=None,
                    target_features_list=None,
                    save_model=False,
                    Y_train_scaler=None,
                    Y_train=None,
                    window_size=None):
    """
    Execute predictions function for a specific flight route
    :param flight_route: current flight route we should train on
    :param test_data_path: the path of test data directory
    :param similarity_score: similarity function
    :param threshold: threshold from the train
    :param random_forest_model: random forest model
    :param X_train_scaler: normalization train input scalar
    :param results_path: the path of results directory
    :param add_plots: indicator whether to add plots or not
    :param run_new_model: indicator whether current flow is new model creation or not
    :param X_train: train input data frame
    :param features_list: the list of features which the user chose for the input
    :param target_features_list: the list of features which the user chose for the target
    :param save_model: indicator whether the user want to save the model or not
    :param Y_train_scaler: normalization train target scalar
    :param Y_train: train target data frame
    :param window_size: window size for each instance in training
    :return: tpr scores, fpr scores, acc scores, delay scores, routes duration
    """

    tpr_scores = defaultdict(list)
    fpr_scores = defaultdict(list)
    acc_scores = defaultdict(list)
    delay_scores = defaultdict(list)
    routes_duration = defaultdict(list)

    # Set a threshold in new model creation flow
    if run_new_model:
        threshold = predict_train_set(random_forest_model,
                                      X_train,
                                      save_model,
                                      add_plots,
                                      threshold,
                                      features_list,
                                      target_features_list,
                                      results_path,
                                      flight_route,
                                      similarity_score,
                                      X_train_scaler,
                                      Y_train,
                                      Y_train_scaler)

    flight_dir = os.path.join(test_data_path, flight_route)
    ATTACKS = get_subdirectories(flight_dir)

    figures_results_path = os.path.join(results_path, "Figures")
    create_directories(figures_results_path)

    attacks_figures_results_path = os.path.join(figures_results_path, "Attacks")
    create_directories(attacks_figures_results_path)

    # Iterate over all attacks in order to find anomalies
    for attack in ATTACKS:
        attack_name = attack

        if "_" in attack_name:
            attack_name = attack_name.split("_")[0]

        current_attack_figures_results_path = os.path.join(attacks_figures_results_path, attack_name)
        create_directories(current_attack_figures_results_path)

        attacks_path = os.path.join(*[str(test_data_path), str(flight_route), str(attack)])
        for flight_csv in os.listdir(f"{attacks_path}"):

            flight_attack_path = os.path.join(*[str(attacks_path), str(flight_csv)])
            df_test_source = pd.read_csv(f"{flight_attack_path}")

            Y_test_labels = df_test_source[[ATTACK_COLUMN]].values
            Y_test_labels_preprocessed = random_forest_model._preprocess(Y_test_labels, Y_test_labels)[1]

            attack_time = len(Y_test_labels)

            input_df_test = df_test_source[features_list]
            target_df_test = df_test_source[target_features_list]

            # Step 1 : Clean test data set
            input_clean_df_test = clean_data(input_df_test)
            target_clean_df_test = clean_data(target_df_test)

            # Step 2: Normalize the data
            X_test = X_train_scaler.transform(input_clean_df_test)

            # Y_test = normalize_data(data=target_clean_df_test,
            #                         scaler="power_transform")[0]

            Y_test = Y_train_scaler.transform(target_clean_df_test)

            Y_test_preprocessed = random_forest_model._preprocess(Y_test, Y_test)[1]

            X_pred = random_forest_model.predict(X_test)
            assert len(Y_test_preprocessed) == len(X_pred)

            scores_test = []
            for i, pred in enumerate(X_pred):
                scores_test.append(anomaly_score(Y_test_preprocessed[i], pred, similarity_score))

            # Add reconstruction error scatter if plots indicator is true
            if add_plots:
                plot_reconstruction_error_scatter(scores=scores_test,
                                                  labels=Y_test_labels_preprocessed,
                                                  threshold=threshold,
                                                  plot_dir=current_attack_figures_results_path,
                                                  title=f'Outlier Score Testing for {flight_csv} in {flight_route}({attack})')

                for i, target_feature in enumerate(target_features_list):
                    title = "Test performance of Random Forest for " + target_feature + " feature in " + flight_csv
                    plot_prediction_performance(Y_train=Y_test_preprocessed[:, i],
                                                X_pred=X_pred[:, i],
                                                results_path=current_attack_figures_results_path,
                                                title=title)

            predictions = [1 if x >= threshold else 0 for x in scores_test]

            # Add roc curve if plots indicator is true
            if add_plots:
                pass
                # plot_roc(y_true=Y_test,y_pred=predictions, plot_dir=results_path,title=f'ROC Curve - {flight_csv} in {flight_route}({attack})')

            attack_start, attack_end = get_attack_boundaries(df_test_source[ATTACK_COLUMN])

            method_scores = get_method_scores(predictions, attack_start, attack_end,
                                              add_window_size=True, window_size=window_size)

            tpr_scores[attack].append(method_scores[0])
            fpr_scores[attack].append(method_scores[1])
            acc_scores[attack].append(method_scores[2])
            delay_scores[attack].append(method_scores[3])
            routes_duration[attack].append(attack_time)

    return tpr_scores, fpr_scores, acc_scores, delay_scores, routes_duration


def predict_train_set(random_forest_model,
                      X_train,
                      save_model,
                      add_plots,
                      threshold,
                      features_list,
                      target_features_list,
                      results_path,
                      flight_route,
                      similarity_score,
                      X_train_scaler,
                      Y_train,
                      Y_train_scaler):
    """
    Execute prediction on the train data set
    :param random_forest_model: random forest model
    :param X_train: train input data frame
    :param save_model: indicator whether the user want to save the model or not
    :param add_plots: indicator whether to add plots or not
    :param threshold: threshold from the train
    :param features_list: the list of features which the user chose for the input
    :param target_features_list: the list of features which the user chose for the target
    :param results_path: the path of results directory
    :param flight_route: current flight route we are working on
    :param similarity_score: similarity function
    :param X_train_scaler: train input normalization scalar
    :param Y_train: train target data frame
    :param Y_train_scaler: train target normalization scalar
    :return: threshold
    """

    X_pred = random_forest_model.predict(X_train)
    scores_train = []

    Y_train_preprocessed = random_forest_model._preprocess(Y_train, Y_train)[1]
    assert len(Y_train_preprocessed) == len(X_pred)

    for i, pred in enumerate(X_pred):
        scores_train.append(anomaly_score(Y_train_preprocessed[i], pred, similarity_score))

    # choose threshold for which <MODEL_THRESHOLD_FROM_TRAINING_PERCENT> % of training were lower
    threshold = get_threshold(scores_train, threshold)

    figures_results_path = os.path.join(results_path, "Figures")
    create_directories(figures_results_path)

    if add_plots:

        train_figures_results_path = os.path.join(figures_results_path, "Train")
        create_directories(train_figures_results_path)

        for i, target_feature in enumerate(target_features_list):
            title = "Training performance of Random Forest for " + target_feature + " in " + flight_route
            plot_prediction_performance(Y_train=Y_train_preprocessed[:, i],
                                        X_pred=X_pred[:, i],
                                        results_path=train_figures_results_path,
                                        title=title)

    # Save created model if the indicator is true
    if save_model:
        data = {}
        data['features'] = features_list
        data['target_features'] = target_features_list
        data['threshold'] = threshold
        data['params'] = get_random_forest_parameters_dictionary()

        model_results_path = os.path.join(results_path, "model_data")
        create_directories(model_results_path)

        model_data_path = os.path.join(*[str(model_results_path), 'model_data.json'])
        with open(f"{model_data_path}", 'w') as outfile:
            json.dump(data, outfile)

        save_model_file_path = os.path.join(model_results_path, flight_route + "_model.pkl")
        with open(save_model_file_path, 'wb') as file:
            pickle.dump(random_forest_model, file)

        save_input_scaler_file_path = os.path.join(model_results_path, flight_route + "_train_scaler.pkl")
        with open(save_input_scaler_file_path, 'wb') as file:
            pickle.dump(X_train_scaler, file)

        save_target_scaler_file_path = os.path.join(model_results_path, flight_route + "_target_scaler.pkl")
        with open(save_target_scaler_file_path, 'wb') as file:
            pickle.dump(Y_train_scaler, file)

    return threshold
